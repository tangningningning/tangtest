"""Adapt saved, decision-time prediction scenarios to optimizer Slot inputs."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from common import project_config
from decision import cache_configuration, task_allocation

from .candidate_evaluator import PredictedSlotInput, SystemModelContext


@dataclass(frozen=True)
class PredictionAdaptationResult:
    """Complete single-sample future path or an explicit unavailable result."""

    is_complete: bool
    forecast_slots: tuple[PredictedSlotInput, ...]
    origin_slot: int
    scenario_index: int
    scenario_id: str
    available_future_horizons: tuple[int, ...]
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def adapt_prediction_origin(
    origin_scenario: Mapping[str, Any],
    model: SystemModelContext,
    *,
    current_slot: int,
    current_time_s: float,
    scenario_index: int,
    prediction_horizon_mask: Any,
) -> PredictionAdaptationResult:
    """Build future Slot inputs from one saved prediction-scenario trajectory.

    ``origin_scenario`` is the output of
    ``decision.cache_configuration.scenario_at_origin`` and is also available as
    ``AlignedRolloutStep.scenario_slice``. ``scenario_index`` selects one
    already-generated common sample path; this function never samples tasks or
    reads future task/trajectory truth. Horizon zero is the current observed Slot
    and is deliberately excluded from the returned future Slots.
    """

    required_keys = (
        "origin_slot",
        "origin_time_s",
        "scenario_ids",
        "horizons",
        "vehicle_ids",
        "vehicle_roles",
        "task_vehicle_ids",
        "service_vehicle_ids",
        "positions_xy",
        "arrival",
        "service_id",
        "input_size",
        "cpu_cycles",
        "rho_hat",
        "metadata",
    )
    missing = [key for key in required_keys if key not in origin_scenario]
    if missing:
        raise ValueError(f"prediction origin is missing required fields: {missing}")

    origin_slot = int(origin_scenario["origin_slot"])
    origin_time_s = float(origin_scenario["origin_time_s"])
    if origin_slot != int(current_slot):
        raise ValueError(
            f"prediction origin {origin_slot} does not match current Slot {current_slot}"
        )
    if not math.isfinite(origin_time_s) or not math.isfinite(float(current_time_s)):
        raise ValueError("prediction origin and current Slot times must be finite")
    if not math.isclose(origin_time_s, float(current_time_s), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("prediction origin time does not match the current Slot time")

    metadata = origin_scenario["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("prediction scenario metadata must be a mapping")
    if isinstance(metadata.get("scenario_seed"), bool) or not isinstance(
        metadata.get("scenario_seed"), (int, np.integer)
    ):
        raise ValueError("prediction scenario metadata lacks its reproducible sample seed")
    boundary = metadata.get("online_information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("prediction artifact lacks its online information boundary")
    if (
        metadata.get("formal_truth_used_for_diagnostics_only") is not True
        or boundary.get("vehicle_prediction") != "observations <= origin_slot - 1"
        or boundary.get("popularity_prediction") != "request_count <= origin_slot - 1"
        or boundary.get("tasks") != "sampled from rho_hat, not rho_star"
        or boundary.get("future_ground_truth_saved") is not False
    ):
        raise ValueError(
            "prediction artifact does not certify prediction-only future inputs"
        )

    scenario_ids = np.asarray(origin_scenario["scenario_ids"]).reshape(-1)
    if (
        isinstance(scenario_index, bool)
        or int(scenario_index) != scenario_index
        or int(scenario_index) < 0
        or int(scenario_index) >= len(scenario_ids)
    ):
        raise ValueError("scenario_index must select an existing prediction sample")
    selected_index = int(scenario_index)
    scenario_id = str(scenario_ids[selected_index])

    horizons = np.asarray(origin_scenario["horizons"], dtype=np.int64).reshape(-1)
    if len(horizons) == 0 or not np.array_equal(
        horizons, np.arange(len(horizons), dtype=np.int64)
    ):
        raise ValueError("prediction horizons must be the ordered offsets 0..H-1")
    horizon_count = len(horizons)
    if prediction_horizon_mask is None:
        raise ValueError("prediction_horizon_mask is required; validity is not inferred")
    valid_mask = cache_configuration.normalize_prediction_horizon_mask(
        prediction_horizon_mask, horizon_count
    )
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid_mask.shape != (horizon_count,):
        raise ValueError("prediction horizon mask does not align with the artifact")
    invalid_indices = np.flatnonzero(~valid_mask)
    valid_end = int(invalid_indices[0]) if len(invalid_indices) else horizon_count
    if np.any(valid_mask[valid_end:]):
        raise ValueError("valid prediction horizons must form a contiguous prefix")

    positions = np.asarray(origin_scenario["positions_xy"], dtype=np.float64)
    arrival = np.asarray(origin_scenario["arrival"])
    service_ids = np.asarray(origin_scenario["service_id"], dtype=np.int64)
    input_bits = np.asarray(origin_scenario["input_size"], dtype=np.float64)
    cpu_cycles = np.asarray(origin_scenario["cpu_cycles"], dtype=np.float64)
    rho_hat = np.asarray(origin_scenario["rho_hat"], dtype=np.float64)
    vehicle_ids = tuple(str(value) for value in np.asarray(origin_scenario["vehicle_ids"]).tolist())
    task_vehicle_ids = tuple(
        str(value) for value in np.asarray(origin_scenario["task_vehicle_ids"]).tolist()
    )
    service_vehicle_ids = tuple(
        str(value) for value in np.asarray(origin_scenario["service_vehicle_ids"]).tolist()
    )
    vehicle_roles = tuple(
        str(value) for value in np.asarray(origin_scenario["vehicle_roles"]).tolist()
    )
    configured_sv_ids: tuple[str, ...] = ()
    if isinstance(model.cache_nodes, Mapping) and (
        "node_ids" in model.cache_nodes and "node_roles" in model.cache_nodes
    ):
        configured_sv_ids = tuple(
            str(node_id)
            for node_id, role in zip(
                model.cache_nodes["node_ids"], model.cache_nodes["node_roles"]
            )
            if str(role).upper() == "SV"
        )

    if arrival.ndim != 3:
        raise ValueError("prediction arrivals must have [sample,horizon,TV] axes")
    sample_count, array_horizons, task_vehicle_count = arrival.shape
    if (
        selected_index >= sample_count
        or sample_count != len(scenario_ids)
        or array_horizons != horizon_count
    ):
        raise ValueError("prediction sample/horizon axes do not match metadata")
    if (
        int(metadata.get("S", -1)) != sample_count
        or int(metadata.get("H_p", -1)) != horizon_count
        or int(metadata.get("F", -1)) != int(model.service_catalog["num_services"])
    ):
        raise ValueError("prediction artifact dimensions conflict with its metadata")
    if any(
        array.shape != arrival.shape
        for array in (service_ids, input_bits, cpu_cycles)
    ):
        raise ValueError("predicted task arrays must share [sample,horizon,TV] axes")
    if (
        positions.ndim != 4
        or positions.shape[:2] != (sample_count, horizon_count)
        or positions.shape[-1] != 2
    ):
        raise ValueError("predicted positions must have [sample,horizon,vehicle,2] axes")
    if rho_hat.shape != (horizon_count, int(model.service_catalog["num_services"])):
        raise ValueError("predicted popularity does not align with the service catalog")
    if (
        len(set(vehicle_ids)) != len(vehicle_ids)
        or len(vehicle_ids) != positions.shape[2]
        or len(set(task_vehicle_ids)) != len(task_vehicle_ids)
        or len(task_vehicle_ids) != task_vehicle_count
        or not set(task_vehicle_ids).issubset(vehicle_ids)
        or not set(service_vehicle_ids).issubset(vehicle_ids)
        or vehicle_ids != service_vehicle_ids + task_vehicle_ids
        or vehicle_roles != ("SV",) * len(service_vehicle_ids) + ("TV",) * len(task_vehicle_ids)
        or (configured_sv_ids and set(configured_sv_ids) != set(service_vehicle_ids))
    ):
        raise ValueError("prediction vehicle IDs/roles do not align with task/position axes")
    if not np.all(np.isin(arrival, [0, 1])):
        raise ValueError("predicted arrivals must be binary")
    if not all(
        np.all(np.isfinite(value))
        for value in (positions, input_bits, cpu_cycles, rho_hat)
    ):
        raise ValueError("prediction arrays contain non-finite values")
    if np.any(rho_hat < 0.0) or not np.allclose(
        rho_hat.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("predicted service popularity must be normalized per horizon")

    catalog = model.service_catalog
    service_count = int(catalog["num_services"])
    mean_input = np.asarray(catalog["mean_input_size_bit"], dtype=np.float64).reshape(-1)
    cycles_per_bit = np.asarray(catalog["cycles_per_bit"], dtype=np.float64).reshape(-1)
    if (
        mean_input.shape != (service_count,)
        or cycles_per_bit.shape != (service_count,)
        or not np.all(np.isfinite(mean_input))
        or not np.all(np.isfinite(cycles_per_bit))
        or np.any(mean_input <= 0.0)
        or np.any(cycles_per_bit <= 0.0)
    ):
        raise ValueError("service catalog lacks valid task workload parameters")

    arrival_probability = float(
        project_config.get_config_value(model.project_cfg, "task.arrival_probability")
    )
    input_jitter = float(
        project_config.get_config_value(model.project_cfg, "task.input_relative_jitter")
    )
    load_jitter = float(
        project_config.get_config_value(model.project_cfg, "task.load_relative_jitter")
    )
    slot_duration_s = float(
        project_config.get_config_value(model.project_cfg, "simulation.dt_s")
    )
    configured_horizon = int(
        project_config.get_config_value(model.project_cfg, "prediction.vehicle.horizon")
    )
    if (
        not math.isfinite(arrival_probability)
        or not 0.0 <= arrival_probability <= 1.0
        or not math.isfinite(input_jitter)
        or not 0.0 <= input_jitter < 1.0
        or not math.isfinite(load_jitter)
        or not 0.0 <= load_jitter < 1.0
        or not math.isfinite(slot_duration_s)
        or slot_duration_s <= 0.0
        or configured_horizon != horizon_count
    ):
        raise ValueError("formal task/Slot prediction parameters are invalid")
    metadata_lambda = float(metadata.get("lambda", float("nan")))
    if not math.isclose(
        metadata_lambda, arrival_probability, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("prediction artifact arrival rate differs from the formal config")
    task_stats = metadata.get("task_stats", {})
    if not isinstance(task_stats, Mapping) or not math.isclose(
        float(task_stats.get("arrival_probability_lambda", float("nan"))),
        arrival_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("prediction task-generation metadata does not match the formal config")

    arrived = arrival.astype(bool)
    if np.any(service_ids[arrived] < 0) or np.any(service_ids[arrived] >= service_count):
        raise ValueError("predicted task service IDs are outside the configured catalog")
    if (
        np.any(service_ids[~arrived] != -1)
        or np.any(input_bits[~arrived] != 0.0)
        or np.any(cpu_cycles[~arrived] != 0.0)
        or np.any(input_bits[arrived] <= 0.0)
        or np.any(cpu_cycles[arrived] <= 0.0)
    ):
        raise ValueError("predicted task arrival/workload fields are inconsistent")
    if np.any(arrived):
        sampled_service = service_ids[arrived]
        input_factor = input_bits[arrived] / mean_input[sampled_service]
        load_factor = cpu_cycles[arrived] / (
            cycles_per_bit[sampled_service] * input_bits[arrived]
        )
        tolerance = 1e-9
        if np.any(input_factor < 1.0 - input_jitter - tolerance) or np.any(
            input_factor > 1.0 + input_jitter + tolerance
        ):
            raise ValueError("predicted task input sizes conflict with catalog/config jitter")
        if np.any(load_factor < 1.0 - load_jitter - tolerance) or np.any(
            load_factor > 1.0 + load_jitter + tolerance
        ):
            raise ValueError("predicted task CPU loads conflict with catalog/config jitter")

    future_horizons = tuple(range(1, valid_end))
    if not future_horizons:
        return PredictionAdaptationResult(
            is_complete=False,
            forecast_slots=(),
            origin_slot=origin_slot,
            scenario_index=selected_index,
            scenario_id=scenario_id,
            available_future_horizons=(),
            failure_reason="no_valid_future_prediction_horizons",
            diagnostics={
                "source_artifact": "prediction_scenarios.npz",
                "horizon_zero_is_current_slot": True,
                "valid_horizon_count": valid_end,
            },
        )

    slots: list[PredictedSlotInput] = []
    safe_metadata_keys = (
        "schema_version",
        "S",
        "H_p",
        "F",
        "scenario_seed",
        "vehicle_observation_seed",
        "lambda",
        "online_information_boundary",
        "same_seed_reproducible",
    )
    safe_metadata = {
        key: copy.deepcopy(metadata[key])
        for key in safe_metadata_keys
        if key in metadata
    }
    safe_metadata["task_generation_lambda"] = arrival_probability
    for horizon_index in future_horizons:
        target_slot = origin_slot + int(horizons[horizon_index])
        target_time_s = origin_time_s + int(horizons[horizon_index]) * slot_duration_s
        remaining_end = valid_end
        remaining_count = remaining_end - horizon_index
        task_list: list[task_allocation.Task] = []
        for task_index, vehicle_id in enumerate(task_vehicle_ids):
            if not bool(arrival[selected_index, horizon_index, task_index]):
                continue
            service_id = int(service_ids[selected_index, horizon_index, task_index])
            task_list.append(
                task_allocation.Task(
                    task_id=(
                        f"predicted:origin={origin_slot}:scenario={scenario_id}:"
                        f"horizon={horizon_index}:vehicle={vehicle_id}"
                    ),
                    vehicle_id=vehicle_id,
                    service_id=service_id,
                    input_bits=float(input_bits[selected_index, horizon_index, task_index]),
                    cpu_cycles=float(cpu_cycles[selected_index, horizon_index, task_index]),
                )
            )

        recentered = copy.deepcopy(dict(origin_scenario))
        # The source artifact also carries aggregate true-vs-predicted distance
        # diagnostics. They are useful for artifact validation but are not
        # decision-time inputs, so they are stripped before handoff.
        recentered_metadata = copy.deepcopy(safe_metadata)
        recentered_metadata["source_artifact_S"] = int(safe_metadata["S"])
        recentered_metadata["source_artifact_H_p"] = int(safe_metadata["H_p"])
        recentered_metadata["S"] = 1
        recentered_metadata["H_p"] = int(remaining_count)
        recentered_metadata["recentered_from_origin_slot"] = int(origin_slot)
        recentered["metadata"] = recentered_metadata
        recentered["origin_slot"] = int(target_slot)
        recentered["origin_time_s"] = float(target_time_s)
        recentered["scenario_ids"] = scenario_ids[selected_index : selected_index + 1].copy()
        recentered["horizons"] = np.arange(remaining_count, dtype=np.int64)
        for key, values in (
            ("positions_xy", positions),
            ("arrival", arrival),
            ("service_id", service_ids),
            ("input_size", input_bits),
            ("cpu_cycles", cpu_cycles),
        ):
            recentered[key] = values[
                selected_index : selected_index + 1,
                horizon_index:remaining_end,
            ].copy()
        recentered["rho_hat"] = rho_hat[horizon_index:remaining_end].copy()
        recentered["prediction_origin_slot"] = int(origin_slot)
        recentered["prediction_origin_time_s"] = float(origin_time_s)
        recentered["prediction_scenario_index"] = selected_index
        recentered["prediction_horizon_index"] = int(horizon_index)

        prediction_metadata = {
            "source_artifact": "prediction_scenarios.npz",
            "prediction_scenario_seed": int(metadata["scenario_seed"]),
            "prediction_origin_slot": int(origin_slot),
            "prediction_origin_time_s": float(origin_time_s),
            "scenario_index": selected_index,
            "scenario_id": scenario_id,
            "original_horizon_index": int(horizon_index),
            "original_horizon_offset": int(horizons[horizon_index]),
            "target_slot": int(target_slot),
            "target_time_s": float(target_time_s),
            "vehicle_prediction_source": "saved vehicle prediction scenarios from current origin",
            "popularity_prediction_source": "saved GRU rho_hat from current origin",
            "task_generation_source": "scenario_generator.sample_tasks using rho_hat",
            "task_arrival_model": "independent Bernoulli per task vehicle per Slot",
            "task_arrival_probability": arrival_probability,
            "service_selection_model": "Categorical(rho_hat at the saved forecast horizon)",
            "input_size_unit": "bit",
            "input_jitter_relative": input_jitter,
            "cpu_cycles_unit": "cycle",
            "load_jitter_relative": load_jitter,
            "vehicle_position_unit": "m",
            "formal_dt_s": slot_duration_s,
            "future_ground_truth_used": False,
            "sample_path_shared_across_candidates": True,
        }
        scenario_id_for_slot = (
            f"prediction_scenarios:origin={origin_slot}:scenario={scenario_id}:"
            f"horizon={horizon_index}:target_slot={target_slot}"
        )
        slots.append(
            PredictedSlotInput(
                slot_id=int(target_slot),
                time_s=float(target_time_s),
                scenario_slice=recentered,
                predicted_tasks=tuple(task_list),
                predicted_vehicle_positions={
                    vehicle_id: positions[selected_index, horizon_index, vehicle_index].copy()
                    for vehicle_index, vehicle_id in enumerate(vehicle_ids)
                },
                prediction_horizon_mask=np.ones(remaining_count, dtype=np.int8),
                scenario_id=scenario_id_for_slot,
                prediction_metadata=prediction_metadata,
            )
        )

    return PredictionAdaptationResult(
        is_complete=True,
        forecast_slots=tuple(slots),
        origin_slot=origin_slot,
        scenario_index=selected_index,
        scenario_id=scenario_id,
        available_future_horizons=future_horizons,
        diagnostics={
            "source_artifact": "prediction_scenarios.npz",
            "scenario_generator_metadata": copy.deepcopy(safe_metadata),
            "horizon_zero_is_current_slot": True,
            "future_ground_truth_used": False,
            "same_sample_path_for_all_future_slots": True,
            "scenario_is_one_prediction_sample_not_an_ensemble_expectation": True,
            "future_slot_count": len(slots),
        },
    )


def adapt_aligned_rollout_step(
    step: Any,
    model: SystemModelContext,
    *,
    scenario_index: int,
) -> PredictionAdaptationResult:
    """Adapt the real ``AlignedRolloutStep`` prediction fields without PPO coupling."""

    return adapt_prediction_origin(
        step.scenario_slice,
        model,
        current_slot=int(step.slot),
        current_time_s=float(step.time_s),
        scenario_index=scenario_index,
        prediction_horizon_mask=step.prediction_horizon_mask,
    )


__all__ = [
    "PredictionAdaptationResult",
    "adapt_aligned_rollout_step",
    "adapt_prediction_origin",
]
