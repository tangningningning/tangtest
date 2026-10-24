"""Shared environment loading, observation, and artifact-alignment interfaces."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import project_config
from decision import (
    cache_configuration,
    slot_environment,
    task_allocation,
    uav_configuration,
)
from scenario.realization import (
    RealizationArtifacts,
    _realization_artifacts_from_directory,
    configured_realization_artifacts,
)
from scenario.vehicle_order import id_axis_indices


NORMALIZED_COST_CONTRACT_VERSION = "node_level_cost_fixed_reference_unweighted_lifecycle_reward_v3"
_REPORTED_COST_NORMALIZATION_CONTRACTS: set[str] = set()


@dataclass(frozen=True)
class EnvironmentContext:
    cfg: dict[str, Any]
    root: Path
    cache_cfg: cache_configuration.CacheOptimizationConfig
    scenarios: dict[str, Any]
    service_catalog: dict[str, Any]
    cache_nodes: dict[str, Any]
    flight_boundary: tuple[float, float, float, float]
    action_spec: ConfigurationActionSpec
    observation_spec: ConfigurationObservationSpec
    initial_cache: np.ndarray
    initial_uav: dict[str, np.ndarray]
    train_origin_start: int
    train_origin_end: int
    data_protocol: str
    episode_slots: int
    arrival_horizon_slots: int
    execution_horizon_slots: int
    terminal_origin: int
    terminal_execution_slot: int
    configured_rollout_steps: int | None
    realization: "RealizationArtifacts | None" = None
    observation_time_to_go: Callable[[int], float] | None = None

def _assert_realization_context_compatible(
    reference: EnvironmentContext,
    candidate: EnvironmentContext,
) -> None:
    """确认新 realization 满足当前环境张量与数据协议。"""

    scalar_fields = (
        "data_protocol",
        "episode_slots",
        "arrival_horizon_slots",
        "execution_horizon_slots",
        "train_origin_start",
        "train_origin_end",
        "terminal_origin",
        "terminal_execution_slot",
        "configured_rollout_steps",
    )
    for field in scalar_fields:
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(
                "Episode realization changes the fixed environment contract: "
                f"field={field}, reference={getattr(reference, field)!r}, "
                f"candidate={getattr(candidate, field)!r}"
            )
    reference_cost_normalization = reference.cfg.get("_meta", {}).get(
        "reward_cost_normalization"
    )
    candidate_cost_normalization = candidate.cfg.get("_meta", {}).get(
        "reward_cost_normalization"
    )
    if reference_cost_normalization != candidate_cost_normalization:
        raise ValueError(
            "Episode realization changes the fixed system reward-cost references: "
            f"reference={reference_cost_normalization!r}, "
            f"candidate={candidate_cost_normalization!r}"
        )
    reference_observation = reference.observation_spec
    candidate_observation = candidate.observation_spec
    if (
        reference_observation.observation_dim
        != candidate_observation.observation_dim
        or reference_observation.base_observation_dim
        != candidate_observation.base_observation_dim
        or reference_observation.feature_slices
        != candidate_observation.feature_slices
        or len(reference_observation.task_vehicle_ids)
        != len(candidate_observation.task_vehicle_ids)
        or len(reference_observation.vehicle_ids)
        != len(candidate_observation.vehicle_ids)
        or reference_observation.service_ids != candidate_observation.service_ids
        or reference_observation.uav_ids != candidate_observation.uav_ids
        or reference_observation.prediction_horizon
        != candidate_observation.prediction_horizon
        or (reference_observation.horizon_mask_slice is None)
        != (candidate_observation.horizon_mask_slice is None)
    ):
        raise ValueError(
            "Episode realization changes the observation tensor contract: "
            f"reference_dim={reference_observation.observation_dim}, "
            f"candidate_dim={candidate_observation.observation_dim}"
        )
    reference_cache_roles = tuple(
        str(value).upper() for value in reference.cache_nodes["node_roles"]
    )
    candidate_cache_roles = tuple(
        str(value).upper() for value in candidate.cache_nodes["node_roles"]
    )
    # Vehicle IDs and their per-Episode order may differ. Each Episode validates
    # its own ID-to-axis map; the shared contract is feature shape and role counts.
    if (
        len(reference.action_spec.cache_node_ids)
        != len(candidate.action_spec.cache_node_ids)
        or reference_cache_roles != candidate_cache_roles
        or reference.action_spec.uav_ids != candidate.action_spec.uav_ids
    ):
        raise ValueError("Episode realization changes Cache node role ordering or UAV ID ordering.")
    if np.asarray(reference.initial_cache).shape != np.asarray(candidate.initial_cache).shape:
        raise ValueError("Episode realization changes the initial cache tensor shape")
    if set(reference.initial_uav) != set(candidate.initial_uav):
        raise ValueError("Episode realization changes the initial UAV ID set")


def _validate_episode_vehicle_mapping(
    realization: RealizationArtifacts,
    scenarios: dict[str, Any],
    *,
    expected_service_vehicle_count: int,
    expected_task_count: int,
    expected_catalog_service_count: int,
    expected_origin_count: int,
    expected_horizon_count: int,
) -> dict[str, Any]:
    """Validate one Episode's ID joins and normalize vehicle-axis arrays by ID."""

    with realization.vehicle_roles.open("r", encoding="utf-8") as stream:
        roles = json.load(stream)
    metadata = scenarios.get("metadata", {})
    if not isinstance(roles, dict) or not isinstance(metadata, dict):
        raise ValueError("vehicle_roles.json and prediction_scenarios metadata must be objects")

    def role_id_list(field: str) -> tuple[str, ...]:
        value = roles.get(field)
        if not isinstance(value, list):
            raise ValueError(f"vehicle_roles.json {field} must be a list of explicit IDs")
        ids = tuple(str(item) for item in value)
        if any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"vehicle_roles.json {field} contains missing or duplicate IDs")
        return ids

    role_service_ids = role_id_list("service_vehicles")
    role_task_ids = role_id_list("task_vehicles")
    role_research_ids = role_id_list("research_vehicles")
    if (
        len(role_service_ids) != int(expected_service_vehicle_count)
        or len(role_task_ids) != int(expected_task_count)
        or len(role_research_ids)
        != int(expected_service_vehicle_count) + int(expected_task_count)
        or set(role_service_ids) & set(role_task_ids)
        or set(role_research_ids) != set(role_service_ids) | set(role_task_ids)
    ):
        raise ValueError("vehicle_roles.json has invalid role counts or overlapping ID sets")

    required_arrays = (
        "origin_slots",
        "origin_times_s",
        "scenario_ids",
        "horizons",
        "vehicle_ids",
        "vehicle_roles",
        "service_vehicle_ids",
        "task_vehicle_ids",
        "positions_xy",
        "arrival",
        "service_id",
        "input_size",
        "cpu_cycles",
        "rho_hat",
    )
    missing = [name for name in required_arrays if name not in scenarios]
    if missing:
        raise ValueError(f"prediction_scenarios.npz is missing required arrays: {missing}")

    scenario_research = tuple(np.asarray(scenarios["vehicle_ids"]).astype(str).tolist())
    scenario_service = tuple(
        np.asarray(scenarios["service_vehicle_ids"]).astype(str).tolist()
    )
    scenario_task = tuple(np.asarray(scenarios["task_vehicle_ids"]).astype(str).tolist())
    scenario_roles = tuple(np.asarray(scenarios["vehicle_roles"]).astype(str).tolist())
    if (
        any(not value.strip() for value in scenario_research + scenario_service + scenario_task)
        or len(set(scenario_research)) != len(scenario_research)
        or len(set(scenario_service)) != len(scenario_service)
        or len(set(scenario_task)) != len(scenario_task)
        or set(scenario_service) & set(scenario_task)
        or set(scenario_research) != set(scenario_service) | set(scenario_task)
        or set(scenario_research) != set(role_research_ids)
        or set(scenario_service) != set(role_service_ids)
        or set(scenario_task) != set(role_task_ids)
        or len(scenario_research)
        != int(expected_service_vehicle_count) + int(expected_task_count)
    ):
        raise ValueError("vehicle_roles.json and prediction_scenarios.npz ID sets do not join")

    metadata_id_fields = {
        "research_vehicle_ids": role_research_ids,
        "SV IDs": role_service_ids,
        "TV IDs": role_task_ids,
    }
    for field, expected_ids in metadata_id_fields.items():
        if field not in metadata:
            continue
        value = metadata[field]
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"prediction_scenarios metadata {field} is not an ID sequence")
        ids = tuple(str(item) for item in value)
        if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
            raise ValueError(f"prediction_scenarios metadata {field} conflicts with role IDs")

    role_order_contract = roles.get("vehicle_order_contract")
    metadata_order_contract = metadata.get("vehicle_order_contract")
    residual_contract = metadata.get("residual_sampling_contract")
    legacy_residual_marker = metadata.get("vehicle_residual_bank_preference")
    residual_provenance = (
        residual_contract or legacy_residual_marker or "UNRECORDED"
    )
    role_by_id = {
        **{vehicle_id: "SV" for vehicle_id in role_service_ids},
        **{vehicle_id: "TV" for vehicle_id in role_task_ids},
    }
    if len(scenario_roles) != len(scenario_research) or any(
        role_by_id.get(vehicle_id) != str(role).upper()
        for vehicle_id, role in zip(scenario_research, scenario_roles, strict=True)
    ):
        raise ValueError("prediction_scenarios.npz vehicle_roles conflicts with its ID mapping")

    # The NPZ vehicle axis is the existing observation order source. Preserve its
    # order within each role and only group the role blocks needed by ObservationSpec.
    primary_service_ids = tuple(
        vehicle_id for vehicle_id in scenario_research if role_by_id[vehicle_id] == "SV"
    )
    primary_task_ids = tuple(
        vehicle_id for vehicle_id in scenario_research if role_by_id[vehicle_id] == "TV"
    )
    primary_research_ids = primary_service_ids + primary_task_ids

    origin_slots = np.asarray(scenarios["origin_slots"])
    origin_times = np.asarray(scenarios["origin_times_s"], dtype=np.float64)
    scenario_ids = np.asarray(scenarios["scenario_ids"])
    horizons = np.asarray(scenarios["horizons"])
    positions = np.asarray(scenarios["positions_xy"])
    if (
        origin_slots.ndim != 1
        or len(origin_slots) != int(expected_origin_count)
        or not np.array_equal(origin_slots, np.arange(len(origin_slots)))
        or origin_times.shape != origin_slots.shape
        or not np.all(np.isfinite(origin_times))
        or np.any(np.diff(origin_times) <= 0.0)
        or scenario_ids.ndim != 1
        or len(scenario_ids) == 0
        or len(set(scenario_ids.astype(str).tolist())) != len(scenario_ids)
        or horizons.ndim != 1
        or len(horizons) != int(expected_horizon_count)
        or len(horizons) == 0
        or horizons.dtype.kind not in "iu"
        or not np.array_equal(
            horizons, np.arange(int(expected_horizon_count), dtype=horizons.dtype)
        )
        or positions.ndim != 5
        or positions.shape[:3]
        != (len(origin_slots), len(scenario_ids), len(horizons))
        or positions.shape[3:] != (len(scenario_research), 2)
    ):
        raise ValueError("prediction_scenarios.npz has invalid origin, horizon, or position axes")
    expected_task_shape = (
        len(origin_slots),
        len(scenario_ids),
        len(horizons),
        len(scenario_task),
    )
    for field in ("arrival", "service_id", "input_size", "cpu_cycles"):
        value = np.asarray(scenarios[field])
        if value.shape != expected_task_shape:
            raise ValueError(
                f"prediction_scenarios.npz {field} shape does not align with TV IDs: "
                f"expected={expected_task_shape}, actual={value.shape}"
            )
    rho_hat = np.asarray(scenarios["rho_hat"])
    if (
        rho_hat.ndim != 3
        or rho_hat.shape[:2] != (len(origin_slots), len(horizons))
        or rho_hat.shape[2] != int(expected_catalog_service_count)
    ):
        raise ValueError("prediction_scenarios.npz rho_hat is not aligned with the service axis")
    for field in (
        "origin_times_s",
        "positions_xy",
        "arrival",
        "service_id",
        "input_size",
        "cpu_cycles",
        "rho_hat",
    ):
        value = np.asarray(scenarios[field])
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError(
                f"prediction_scenarios.npz {field} contains non-finite or non-numeric values"
            )

    vehicle_indices = id_axis_indices(
        scenario_research,
        primary_research_ids,
        context="prediction_scenarios.npz positions_xy",
    )
    task_indices = id_axis_indices(
        scenario_task,
        primary_task_ids,
        context="prediction_scenarios.npz task arrays",
    )
    normalized = dict(scenarios)
    normalized["vehicle_ids"] = np.asarray(primary_research_ids, dtype=str)
    normalized["vehicle_roles"] = np.asarray(
        ("SV",) * len(primary_service_ids) + ("TV",) * len(primary_task_ids),
        dtype=str,
    )
    normalized["service_vehicle_ids"] = np.asarray(primary_service_ids, dtype=str)
    normalized["task_vehicle_ids"] = np.asarray(primary_task_ids, dtype=str)
    normalized["positions_xy"] = np.take(positions, vehicle_indices, axis=-2).copy()
    for field in ("arrival", "service_id", "input_size", "cpu_cycles"):
        normalized[field] = np.take(np.asarray(scenarios[field]), task_indices, axis=-1).copy()
    normalized_metadata = dict(metadata)
    normalized_metadata["loader_compatibility"] = {
        "observation_vehicle_order_source": (
            "prediction_scenarios.npz:vehicle_ids; stable subsequence within each role"
        ),
        "source_vehicle_order_markers": {
            "vehicle_roles": role_order_contract,
            "prediction_scenarios": metadata_order_contract,
        },
        "source_residual_sampling_provenance": residual_provenance,
        "saved_prediction_scenarios_reused": True,
        "scenario_metadata_preserved": True,
    }
    normalized["metadata"] = normalized_metadata
    return normalized


def load_environment_context(
    config_path: str = "configs/dqn/online_joint_dqn.toml",
    realization_dir: str | Path | None = None,
) -> EnvironmentContext:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    configured_cost_contract = project_config.get_config_value(
        cfg, "objective.cost_normalization.contract_version"
    )
    if (
        configured_cost_contract is not None
        and configured_cost_contract != NORMALIZED_COST_CONTRACT_VERSION
    ):
        raise ValueError(
            "Unsupported objective.cost_normalization.contract_version: "
            f"{configured_cost_contract!r}"
        )
    data_protocol = project_config.get_episode_data_protocol(cfg)
    root = Path(cfg["_meta"]["project_root"])
    realization = (
        configured_realization_artifacts(cfg, root)
        if realization_dir is None
        else _realization_artifacts_from_directory(
            Path(realization_dir)
            if Path(realization_dir).is_absolute()
            else root / Path(realization_dir)
        )
    )
    cache_cfg = cache_configuration.cache_config_from_project(cfg)
    scenarios = cache_configuration.load_prediction_scenarios(
        realization.prediction_scenarios
    )
    scenario_origin_count = int(scenarios["origin_slots"].shape[0])
    required_execution_origins = int(data_protocol["execution_horizon_slots"])
    if scenario_origin_count < required_execution_origins:
        raise ValueError(
            "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
            f"prediction_scenarios has {scenario_origin_count} origins, "
            f"but the complete execution contract requires "
            f"{required_execution_origins} (slots 0..{required_execution_origins - 1})."
        )
    if scenario_origin_count != required_execution_origins:
        raise ValueError(
            "execution scenario origin count differs from the configured contract: "
            f"found={scenario_origin_count}, required={required_execution_origins}"
        )
    scenarios = _validate_episode_vehicle_mapping(
        realization,
        scenarios,
        expected_service_vehicle_count=int(
            project_config.get_config_value(cfg, "simulation.num_service_vehicles")
        ),
        expected_task_count=int(
            project_config.get_config_value(cfg, "simulation.num_task_vehicles")
        ),
        expected_catalog_service_count=int(
            project_config.get_config_value(cfg, "service.count")
        ),
        expected_origin_count=int(data_protocol["execution_horizon_slots"]),
        expected_horizon_count=int(
            project_config.get_config_value(cfg, "prediction.vehicle.horizon")
        ),
    )
    service_catalog = cache_configuration.load_service_catalog(
        project_config.resolve_path(root, project_config.get_config_value(cfg, "paths.service_catalog"))
    )
    train_start = int(data_protocol["train_origin_start"])
    train_end = int(data_protocol["train_origin_end"])
    origin_count = int(scenarios["origin_slots"].shape[0])
    if train_start < 0 or train_end < train_start or train_end >= origin_count:
        raise ValueError("训练起点范围超出 prediction_scenarios。")
    first_origin = cache_configuration.scenario_at_origin(scenarios, train_start)
    cache_nodes = cache_configuration.build_cache_nodes(first_origin, cache_cfg)
    expected_cache_node_count = (
        int(project_config.get_config_value(cfg, "simulation.num_service_vehicles"))
        + int(project_config.get_config_value(cfg, "nodes.rsu.count"))
        + int(project_config.get_config_value(cfg, "nodes.uav.count"))
    )
    if len(cache_nodes["node_ids"]) != expected_cache_node_count:
        raise ValueError(
            "Cache 节点数与固定系统配置不一致："
            f"actual={len(cache_nodes['node_ids'])}, "
            f"expected={expected_cache_node_count}"
        )
    cache_cost_reference = cache_configuration.initialize_cache_cost_normalization(
        cache_nodes,
        service_catalog["service_sizes_mb"],
        float(
            project_config.get_config_value(
                cfg, "objective.cost_normalization.cache_reference_s_per_node"
            )
        ),
    )
    movement_reference_by_uav = uav_configuration.movement_energy_upper_bounds_by_uav_j(
        cfg
    )
    movement_reference_j = float(sum(movement_reference_by_uav.values()))
    beta_ca = float(project_config.get_config_value(cfg, "objective.beta_ca"))
    beta_q = float(project_config.get_config_value(cfg, "objective.beta_q"))
    movement_cost_reference = {
        "reference_j": movement_reference_j,
        "reference_j_by_uav": movement_reference_by_uav,
        "reference_kind": (
            "per_uav_conservative_speed_limit_upper_bound"
            if movement_reference_j > 0.0
            else "zero_no_possible_positive_movement_energy"
        ),
        "formula": "E_mov_u_ref_j=xi_mov_u_J_per_m * vmax_u_m_per_s * slot_duration_s",
        "uav_count": int(project_config.get_config_value(cfg, "nodes.uav.count")),
        "unit": "J",
    }
    cfg["_meta"]["reward_cost_normalization"] = {
        "contract_version": NORMALIZED_COST_CONTRACT_VERSION,
        "cache": cache_cost_reference,
        "movement": movement_cost_reference,
        "legacy_system_utility_beta_config": {
            "beta_ca": beta_ca,
            "beta_q": beta_q,
            "source": (
                "retained compatibility/other aggregation fields; not applied by "
                "Online Joint DQN lifecycle_system_reward"
            ),
        },
    }
    reference_record = json.dumps(
        cfg["_meta"]["reward_cost_normalization"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if reference_record not in _REPORTED_COST_NORMALIZATION_CONTRACTS:
        print(
            "[reward-cost normalization] "
            "C_ca=sum_m(T_ca,m_s/T_ca,m_ref_s), C_q=sum_u(E_mov_u_J/E_mov_u_ref_J); "
            f"T_ca_m_ref_s_by_node="
            f"{dict(zip(cache_nodes['node_ids'], cache_cost_reference['reference_s_by_node']))} "
            f"(sum={cache_cost_reference['reference_s_total']:.12g} s) "
            f"({cache_cost_reference['reference_kind']}), "
            f"E_mov_u_ref_j_by_uav={movement_reference_by_uav} "
            f"(sum={movement_cost_reference['reference_j']:.12g} J) "
            f"({movement_cost_reference['reference_kind']}), "
            f"legacy_beta_ca_config_not_applied={beta_ca:.12g}, "
            f"legacy_beta_q_config_not_applied={beta_q:.12g}",
            flush=True,
        )
        _REPORTED_COST_NORMALIZATION_CONTRACTS.add(reference_record)
    action_spec = ConfigurationActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    roles = [str(value).upper() for value in cache_nodes["node_roles"]]
    if (
        not action_spec.cache_node_ids
        or len(set(action_spec.cache_node_ids)) != len(action_spec.cache_node_ids)
        or len(roles) != len(action_spec.cache_node_ids)
        or any(role not in {"SV", "RSU", "UAV"} for role in roles)
        or not action_spec.uav_ids
    ):
        raise ValueError(
            "Environment requires unique ordered Cache nodes with supported roles and at least one UAV."
        )
    prediction_horizon = int(
        project_config.get_config_value(cfg, "prediction.vehicle.horizon")
    )
    observation_spec = ConfigurationObservationSpec.from_cache_nodes(
        cache_nodes,
        service_catalog["service_ids"].tolist(),
        action_spec.uav_ids,
        [str(value) for value in first_origin["vehicle_ids"].tolist()],
        task_vehicle_ids=[str(value) for value in first_origin["task_vehicle_ids"].tolist()],
        include_time_to_go=True,
        prediction_horizon=prediction_horizon,
        include_prediction_horizon_mask=True,
    )
    initial_cache = cache_configuration.popularity_aware_initial_cache(
        service_catalog,
        cache_nodes,
        cache_cfg,
    )
    initial_uav = uav_configuration.initial_uav_positions_from_project(cfg)
    return EnvironmentContext(
        cfg=cfg,
        root=root,
        cache_cfg=cache_cfg,
        scenarios=scenarios,
        service_catalog=service_catalog,
        cache_nodes=cache_nodes,
        flight_boundary=uav_configuration.boundary_from_project(cfg),
        action_spec=action_spec,
        observation_spec=observation_spec,
        initial_cache=initial_cache,
        initial_uav=initial_uav,
        train_origin_start=train_start,
        train_origin_end=train_end,
        data_protocol=str(data_protocol["protocol"]),
        episode_slots=int(data_protocol["episode_slots"]),
        arrival_horizon_slots=int(data_protocol["arrival_horizon_slots"]),
        execution_horizon_slots=int(data_protocol["execution_horizon_slots"]),
        terminal_origin=int(data_protocol["terminal_origin"]),
        terminal_execution_slot=int(data_protocol["terminal_execution_slot"]),
        configured_rollout_steps=(
            None
            if data_protocol["rollout_steps"] is None
            else int(data_protocol["rollout_steps"])
        ),
        realization=realization,
    )

def aligned_train_steps(
    context: EnvironmentContext,
    origin_start: int | None = None,
    max_steps: int | None = None,
    *,
    include_bootstrap_step: bool = False,
) -> list[Any]:
    start = context.train_origin_start if origin_start is None else int(origin_start)
    if start < context.train_origin_start or start > context.train_origin_end:
        raise ValueError("origin_start 必须位于 formal train origins 内。")
    available_steps = context.train_origin_end - start + 1
    step_count = available_steps
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps 必须大于0。")
        step_count = min(available_steps, int(max_steps))
    if include_bootstrap_step and start + step_count > context.train_origin_end:
        raise ValueError("train terminal 后不允许读取 test origin 作为 bootstrap step。")
    return prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        start,
        step_count,
        (
            context.realization.vehicle_predictions
            if context.realization is not None
            else project_config.resolve_path(
                context.root,
                project_config.get_config_value(
                    context.cfg,
                    "paths.vehicle_predictions",
                ),
            )
        ),
        (
            context.realization.popularity_predictions
            if context.realization is not None
            else project_config.resolve_path(
                context.root,
                project_config.get_config_value(
                    context.cfg,
                    "paths.popularity_predictions",
                ),
            )
        ),
        (
            context.realization.tasks
            if context.realization is not None
            else project_config.resolve_path(
                context.root,
                project_config.get_config_value(context.cfg, "paths.tasks"),
            )
        ),
        (
            context.realization.trajectory
            if context.realization is not None
            else project_config.resolve_path(
                context.root,
                project_config.get_config_value(context.cfg, "paths.trajectory"),
            )
        ),
        include_bootstrap_step=bool(include_bootstrap_step),
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(
            project_config.get_config_value(context.cfg, "simulation.dt_s")
        ),
        prediction_terminal_origin=int(context.train_origin_end),
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_task_vehicle_ids=context.observation_spec.task_vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(
            context.observation_spec.horizon_mask_slice is not None
        ),
    )


def aligned_execution_steps(context: EnvironmentContext) -> list[Any]:
    """Load all causal decision inputs for a complete, extended Training Episode."""

    if context.realization is None:
        raise ValueError("完整生命周期执行要求显式的 Training realization。")
    if (
        context.train_origin_start != 0
        or context.train_origin_end != context.arrival_horizon_slots - 1
    ):
        raise ValueError(
            "完整生命周期执行要求任务到达 origins 为零基 0..T-1。"
        )
    available_origins = int(context.scenarios["origin_slots"].shape[0])
    if available_origins < context.execution_horizon_slots:
        raise ValueError(
            "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
            f"need {context.execution_horizon_slots} decision origins, "
            f"found {available_origins}"
        )
    return prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        0,
        context.execution_horizon_slots,
        context.realization.vehicle_predictions,
        context.realization.popularity_predictions,
        context.realization.tasks,
        context.realization.trajectory,
        include_bootstrap_step=False,
        prediction_horizon=int(context.observation_spec.prediction_horizon),
        prediction_dt_s=float(
            project_config.get_config_value(context.cfg, "simulation.dt_s")
        ),
        prediction_terminal_origin=context.terminal_execution_slot,
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_task_vehicle_ids=context.observation_spec.task_vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(
            context.observation_spec.horizon_mask_slice is not None
        ),
        arrival_horizon_slots=context.arrival_horizon_slots,
        require_full_step_count=True,
    )


@dataclass(frozen=True)
class ConfigurationActionSpec:
    """提供动态 Cache 节点和 UAV 的固定顺序。"""

    cache_node_ids: tuple[str, ...]
    uav_ids: tuple[str, ...]

    @classmethod
    def from_cache_nodes_and_project(
        cls,
        cache_nodes: dict[str, Any],
        project_cfg: dict[str, Any],
    ) -> "ConfigurationActionSpec":
        cache_node_ids = tuple(str(value) for value in cache_nodes["node_ids"])
        uav_ids = tuple(
            str(item["id"])
            for item in project_config.get_config_value(project_cfg, "nodes.uav.instances", [])
        )
        if not cache_node_ids:
            raise ValueError("正式控制至少需要一个动态 Cache 节点。")
        if not uav_ids:
            raise ValueError("正式控制至少需要一个 UAV。")
        return cls(cache_node_ids=cache_node_ids, uav_ids=uav_ids)


@dataclass(frozen=True)
class ConfigurationObservationSpec:
    """Configuration observation layout shared by the environment and DQN."""

    cache_node_ids: tuple[str, ...]
    service_ids: tuple[int, ...]
    uav_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    task_vehicle_ids: tuple[str, ...]
    cache_slice: slice
    uav_position_slice: slice
    predicted_vehicle_slice: slice
    popularity_slice: slice
    horizon_mask_slice: slice | None
    time_to_go_slice: slice | None
    base_observation_dim: int
    # Preserve these None-only metadata keys for existing feature-slice contracts.
    edge_service_lock_slice: None
    edge_upload_workload_slice: None
    edge_compute_workload_slice: None
    cloud_backlog_slice: None
    uav_tv_upload_slice: None
    observation_dim: int
    prediction_horizon: int = 1

    @property
    def feature_slices(self) -> dict[str, tuple[int, int] | None]:
        """Return the named feature offsets that define this observation schema."""

        slices = {
            "cache_state": self.cache_slice,
            "uav_positions": self.uav_position_slice,
            "predicted_vehicle_positions": self.predicted_vehicle_slice,
            "predicted_service_popularity": self.popularity_slice,
            "prediction_horizon_mask": self.horizon_mask_slice,
            "time_to_go": self.time_to_go_slice,
            "edge_service_locks": self.edge_service_lock_slice,
            "edge_upload_workload": self.edge_upload_workload_slice,
            "edge_remaining_cpu_workload": self.edge_compute_workload_slice,
            "cloud_backlog": self.cloud_backlog_slice,
            "uav_tv_upload_workload": self.uav_tv_upload_slice,
        }
        return {
            name: None if value is None else (int(value.start), int(value.stop))
            for name, value in slices.items()
        }

    @classmethod
    def create(
        cls,
        cache_node_ids: list[str] | tuple[str, ...],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
        *,
        task_vehicle_ids: list[str] | tuple[str, ...],
        include_time_to_go: bool = False,
        prediction_horizon: int = 1,
        include_prediction_horizon_mask: bool = False,
    ) -> "ConfigurationObservationSpec":
        cache_node_ids_t = tuple(str(value) for value in cache_node_ids)
        service_ids_t = tuple(int(value) for value in service_ids)
        uav_ids_t = tuple(str(value) for value in uav_ids)
        vehicle_ids_t = tuple(str(value) for value in vehicle_ids)
        task_vehicle_ids_t = tuple(str(value) for value in task_vehicle_ids)
        if len(set(cache_node_ids_t)) != len(cache_node_ids_t) or not cache_node_ids_t:
            raise ValueError("cache_node_ids 不能为空且不能重复。")
        if service_ids_t != tuple(range(len(service_ids_t))):
            raise ValueError("service_ids 必须从 0 连续排列。")
        if len(set(uav_ids_t)) != len(uav_ids_t) or not uav_ids_t:
            raise ValueError("uav_ids 不能为空且不能重复。")
        if len(set(vehicle_ids_t)) != len(vehicle_ids_t) or not vehicle_ids_t:
            raise ValueError("vehicle_ids 不能为空且不能重复。")
        if (
            len(set(task_vehicle_ids_t)) != len(task_vehicle_ids_t)
            or not task_vehicle_ids_t
            or not set(task_vehicle_ids_t).issubset(vehicle_ids_t)
        ):
            raise ValueError("task_vehicle_ids 必须唯一、非空且属于 vehicle_ids。")
        prediction_horizon_i = int(prediction_horizon)
        if prediction_horizon_i <= 0:
            raise ValueError("prediction_horizon 必须大于0。")
        if include_prediction_horizon_mask and prediction_horizon_i <= 1:
            raise ValueError("prediction_horizon mask 需要 prediction_horizon > 1。")

        start = 0
        cache_len = len(cache_node_ids_t) * len(service_ids_t)
        cache_slice = slice(start, start + cache_len)
        start += cache_len
        uav_len = len(uav_ids_t) * 2
        uav_position_slice = slice(start, start + uav_len)
        start += uav_len
        vehicle_len = len(vehicle_ids_t) * 2 * prediction_horizon_i
        predicted_vehicle_slice = slice(start, start + vehicle_len)
        start += vehicle_len
        popularity_len = len(service_ids_t) * prediction_horizon_i
        popularity_slice = slice(start, start + popularity_len)
        start += popularity_len
        horizon_mask_slice = None
        if include_prediction_horizon_mask:
            horizon_mask_slice = slice(start, start + prediction_horizon_i)
            start += prediction_horizon_i
        time_to_go_slice = None
        if include_time_to_go:
            time_to_go_slice = slice(start, start + 1)
            start += 1
        base_observation_dim = start
        edge_service_lock_slice = None
        edge_upload_workload_slice = None
        edge_compute_workload_slice = None
        cloud_backlog_slice = None
        uav_tv_upload_slice = None
        return cls(
            cache_node_ids=cache_node_ids_t,
            service_ids=service_ids_t,
            uav_ids=uav_ids_t,
            vehicle_ids=vehicle_ids_t,
            task_vehicle_ids=task_vehicle_ids_t,
            cache_slice=cache_slice,
            uav_position_slice=uav_position_slice,
            predicted_vehicle_slice=predicted_vehicle_slice,
            popularity_slice=popularity_slice,
            horizon_mask_slice=horizon_mask_slice,
            time_to_go_slice=time_to_go_slice,
            base_observation_dim=base_observation_dim,
            edge_service_lock_slice=edge_service_lock_slice,
            edge_upload_workload_slice=edge_upload_workload_slice,
            edge_compute_workload_slice=edge_compute_workload_slice,
            cloud_backlog_slice=cloud_backlog_slice,
            uav_tv_upload_slice=uav_tv_upload_slice,
            observation_dim=start,
            prediction_horizon=prediction_horizon_i,
        )

    @classmethod
    def from_cache_nodes(
        cls,
        cache_nodes: dict[str, Any],
        service_ids: list[int] | tuple[int, ...],
        uav_ids: list[str] | tuple[str, ...],
        vehicle_ids: list[str] | tuple[str, ...],
        *,
        task_vehicle_ids: list[str] | tuple[str, ...],
        include_time_to_go: bool = False,
        prediction_horizon: int = 1,
        include_prediction_horizon_mask: bool = False,
    ) -> "ConfigurationObservationSpec":
        return cls.create(
            cache_nodes["node_ids"],
            service_ids,
            uav_ids,
            vehicle_ids,
            task_vehicle_ids=task_vehicle_ids,
            include_time_to_go=include_time_to_go,
            prediction_horizon=prediction_horizon,
            include_prediction_horizon_mask=include_prediction_horizon_mask,
        )


def _normalize_xy(
    xy: Any,
    boundary: tuple[float, float, float, float],
    label: str,
) -> np.ndarray:
    x_min, y_min, x_max, y_max = [float(value) for value in boundary]
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("flight_boundary 非法。")
    array = np.asarray(xy, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{label} 坐标 shape={array.shape}，期望 (2,)。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 坐标包含 NaN/Inf。")
    normalized = np.asarray(
        [
            (float(array[0]) - x_min) / (x_max - x_min),
            (float(array[1]) - y_min) / (y_max - y_min),
        ],
        dtype=np.float64,
    )
    if np.any(normalized < -1e-6) or np.any(normalized > 1.0 + 1e-6):
        raise ValueError(f"{label} 归一化坐标越界：{normalized.tolist()}。")
    return np.clip(normalized, 0.0, 1.0)


def _ordered_position_array(
    positions: dict[str, Any] | Any,
    ids: tuple[str, ...],
    name: str,
) -> np.ndarray:
    if isinstance(positions, dict):
        if set(str(key) for key in positions) != set(ids):
            raise ValueError(f"{name} IDs 与 observation spec 不一致。")
        return np.asarray([positions[vehicle_id] for vehicle_id in ids], dtype=np.float64)
    array = np.asarray(positions, dtype=np.float64)
    if array.shape != (len(ids), 2):
        raise ValueError(f"{name} shape={array.shape}，期望 {(len(ids), 2)}。")
    return array


def _ordered_popularity_array(
    popularity: dict[int | str, float] | Any,
    service_ids: tuple[int, ...],
) -> np.ndarray:
    if isinstance(popularity, dict):
        normalized_dict = {int(key): float(value) for key, value in popularity.items()}
        if set(normalized_dict) != set(service_ids):
            raise ValueError("predicted_service_popularity service IDs 与 spec 不一致。")
        values = np.asarray(
            [normalized_dict[service_id] for service_id in service_ids],
            dtype=np.float64,
        )
    else:
        values = np.asarray(popularity, dtype=np.float64).reshape(-1)
        if values.shape != (len(service_ids),):
            raise ValueError(
                f"predicted_service_popularity shape={values.shape}，期望 {(len(service_ids),)}。"
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("predicted_service_popularity 包含 NaN/Inf。")
    if np.any(values < -1e-8):
        raise ValueError("predicted_service_popularity 不能为负。")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("predicted_service_popularity 必须已经归一化为概率分布。")
    return values.astype(np.float64)


def _prediction_horizon_mask(
    observation_spec: ConfigurationObservationSpec,
    value: Any,
) -> np.ndarray:
    horizon = int(observation_spec.prediction_horizon)
    if observation_spec.horizon_mask_slice is None:
        if value is not None:
            raise ValueError("v1/v2 observation 不接受 prediction horizon mask。")
        return np.ones(horizon, dtype=bool)
    if value is None:
        raise ValueError("v3 observation requires prediction horizon mask")
    mask = np.asarray(value, dtype=np.int8).reshape(-1)
    if mask.shape != (horizon,) or not np.all(np.isin(mask, [0, 1])):
        raise ValueError(f"prediction_horizon_mask shape/value 非法：{mask}")
    return mask.astype(bool)


def _ordered_vehicle_horizon_array(
    positions: dict[str, Any] | Any,
    ids: tuple[str, ...],
    horizon: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if horizon == 1 and isinstance(positions, dict):
        return _ordered_position_array(positions, ids, "predicted_vehicle_positions")[None, ...]
    try:
        sequence = list(positions)
    except TypeError as exc:
        raise ValueError("predicted_vehicle_positions horizon 输入不可迭代。") from exc
    if len(sequence) != horizon:
        raise ValueError(
            f"predicted_vehicle_positions horizon 数量={len(sequence)}，期望 {horizon}。"
        )
    arrays: list[np.ndarray] = []
    for horizon_index, item in enumerate(sequence):
        if valid_mask[horizon_index]:
            arrays.append(
                _ordered_position_array(
                    item,
                    ids,
                    f"predicted_vehicle_positions[h={horizon_index}]",
                )
            )
        else:
            arrays.append(np.zeros((len(ids), 2), dtype=np.float64))
    return np.stack(arrays, axis=0)


def _ordered_popularity_horizon_array(
    popularity: dict[int | str, float] | Any,
    service_ids: tuple[int, ...],
    horizon: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if horizon == 1 and isinstance(popularity, dict):
        return _ordered_popularity_array(popularity, service_ids)[None, ...]
    try:
        sequence = list(popularity)
    except TypeError as exc:
        raise ValueError("predicted_service_popularity horizon 输入不可迭代。") from exc
    if len(sequence) != horizon:
        raise ValueError(
            f"predicted_service_popularity horizon 数量={len(sequence)}，期望 {horizon}。"
        )
    arrays: list[np.ndarray] = []
    for horizon_index, item in enumerate(sequence):
        if valid_mask[horizon_index]:
            arrays.append(
                _ordered_popularity_array(
                    item,
                    service_ids,
                )
            )
        else:
            arrays.append(np.zeros(len(service_ids), dtype=np.float64))
    return np.stack(arrays, axis=0)


def normalized_time_to_go(
    origin_index: int,
    train_origin_start: int,
    train_origin_end: int,
) -> float:
    """Return the finite-horizon state feature (end - origin) / (end - start)."""

    start = int(train_origin_start)
    end = int(train_origin_end)
    origin = int(origin_index)
    if end <= start:
        raise ValueError("time_to_go requires train_origin_end > train_origin_start")
    if origin < start or origin > end:
        raise ValueError("time_to_go origin is outside the formal train range")
    value = (end - origin) / (end - start)
    if not 0.0 <= value <= 1.0:
        raise AssertionError("time_to_go must be within [0, 1]")
    return float(value)


def prediction_horizon_valid_mask(
    origin_index: int,
    prediction_horizon: int,
    prediction_terminal_origin: int,
) -> np.ndarray:
    """Return the H-step validity mask without reading any prediction data."""

    origin = int(origin_index)
    horizon = int(prediction_horizon)
    terminal = int(prediction_terminal_origin)
    if horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if origin < 0 or terminal < origin:
        raise ValueError("prediction horizon origin/terminal bounds are invalid")
    return np.asarray(
        [int(origin + horizon_index <= terminal) for horizon_index in range(horizon)],
        dtype=np.int8,
    )


def build_configuration_observation(
    previous_cache_state: Any,
    previous_uav_positions: dict[str, Any] | Any,
    predicted_vehicle_positions: dict[str, Any] | Any,
    predicted_service_popularity: dict[int | str, float] | Any,
    observation_spec: ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
    *,
    time_to_go: float | None = None,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Build the formal state from current configuration and predictions."""

    m = len(observation_spec.cache_node_ids)
    f = len(observation_spec.service_ids)
    previous_cache = np.asarray(previous_cache_state, dtype=np.int8)
    if previous_cache.shape != (m, f):
        raise ValueError(f"previous_cache_state shape={previous_cache.shape}，期望 {(m, f)}。")
    if not np.all(np.isin(previous_cache, [0, 1])):
        raise ValueError("previous_cache_state 必须只包含 0/1。")

    uav_positions = _ordered_position_array(
        previous_uav_positions,
        observation_spec.uav_ids,
        "previous_uav_positions",
    )
    valid_horizon_mask = _prediction_horizon_mask(
        observation_spec,
        prediction_horizon_mask,
    )
    vehicle_positions = _ordered_vehicle_horizon_array(
        predicted_vehicle_positions,
        observation_spec.vehicle_ids,
        int(observation_spec.prediction_horizon),
        valid_horizon_mask,
    )
    popularity = _ordered_popularity_horizon_array(
        predicted_service_popularity,
        observation_spec.service_ids,
        int(observation_spec.prediction_horizon),
        valid_horizon_mask,
    )
    cache_features = previous_cache.reshape(-1).astype(np.float64)
    uav_features = np.concatenate(
        [
            _normalize_xy(uav_positions[index], flight_boundary, f"UAV {uav_id}")
            for index, uav_id in enumerate(observation_spec.uav_ids)
        ]
    )
    vehicle_feature_parts: list[np.ndarray] = []
    for horizon_index in range(int(observation_spec.prediction_horizon)):
        if not bool(valid_horizon_mask[horizon_index]):
            vehicle_feature_parts.append(
                np.zeros(len(observation_spec.vehicle_ids) * 2, dtype=np.float64)
            )
            continue
        vehicle_feature_parts.append(
            np.concatenate(
                [
                    _normalize_xy(
                        vehicle_positions[horizon_index, index],
                        flight_boundary,
                        f"vehicle {vehicle_id} h={horizon_index}",
                    )
                    for index, vehicle_id in enumerate(observation_spec.vehicle_ids)
                ]
            )
        )
    vehicle_features = np.concatenate(vehicle_feature_parts)
    popularity_features = popularity.reshape(-1)
    horizon_mask_features = (
        valid_horizon_mask.astype(np.float64)
        if observation_spec.horizon_mask_slice is not None
        else np.empty(0, dtype=np.float64)
    )
    if observation_spec.time_to_go_slice is None:
        if time_to_go is not None:
            raise ValueError("v1 observation must not receive time_to_go")
        time_features = np.empty(0, dtype=np.float64)
    else:
        if time_to_go is None:
            raise ValueError("v2 observation requires time_to_go")
        time_value = float(time_to_go)
        if not np.isfinite(time_value) or not 0.0 <= time_value <= 1.0:
            raise ValueError("time_to_go must be finite and within [0, 1]")
        time_features = np.asarray([time_value], dtype=np.float64)

    observation = np.concatenate(
        [
            cache_features,
            uav_features,
            vehicle_features,
            popularity_features,
            horizon_mask_features,
            time_features,
        ]
    )
    if observation.shape != (observation_spec.observation_dim,):
        raise AssertionError("observation length 与 spec 不一致。")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation 包含 NaN/Inf。")
    return observation.astype(np.float32)


def build_step_observation(
    previous_cache_state: Any,
    previous_uav_positions: dict[str, Any] | Any,
    step: Any,
    observation_spec: ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
    train_origin_start: int,
    train_origin_end: int,
    *,
    time_to_go_override: float | None = None,
) -> np.ndarray:
    """构造由当前配置和预测信息组成的 Slot 观测。"""

    time_to_go = (
        float(time_to_go_override)
        if time_to_go_override is not None
        else normalized_time_to_go(
            step.origin_index,
            train_origin_start,
            train_origin_end,
        )
        if observation_spec.time_to_go_slice is not None
        else None
    )
    if int(observation_spec.prediction_horizon) > 1:
        predicted_vehicle_positions = step.predicted_vehicle_positions_horizon
        predicted_service_popularity = step.predicted_service_popularity_horizon
        prediction_horizon_mask = step.prediction_horizon_mask
    else:
        predicted_vehicle_positions = step.predicted_vehicle_positions
        predicted_service_popularity = step.predicted_service_popularity
        prediction_horizon_mask = None
    return build_configuration_observation(
        previous_cache_state,
        previous_uav_positions,
        predicted_vehicle_positions,
        predicted_service_popularity,
        observation_spec,
        flight_boundary,
        time_to_go=time_to_go,
        prediction_horizon_mask=prediction_horizon_mask,
    )




def _copy_uav_positions(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(uav_id): (float(np.asarray(xy)[0]), float(np.asarray(xy)[1]))
        for uav_id, xy in positions.items()
    }


@dataclass(frozen=True)
class AlignedRolloutStep:
    """已经按同一个真实 slot/time 对齐的一步输入。"""

    origin_index: int
    slot: int
    time_s: float
    scenario_slice: dict[str, Any]
    predicted_vehicle_positions: dict[str, np.ndarray]
    predicted_service_popularity: dict[int, float]
    predicted_vehicle_positions_horizon: tuple[dict[str, np.ndarray], ...]
    predicted_service_popularity_horizon: tuple[dict[int, float], ...]
    prediction_horizon_mask: np.ndarray | None
    actual_tasks: list[task_allocation.Task]
    actual_vehicle_positions: dict[str, np.ndarray]


def _load_vehicle_prediction_h0(
    path: Path,
    allowed_origins: set[int] | None = None,
    expected_vehicle_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[int, dict[str, tuple[float, float, int, float]]]:
    by_origin: dict[int, dict[str, tuple[float, float, int, float]]] = {}
    order_by_origin: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["horizon"]) != 0:
                continue
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
            cutoff = int(row["observation_cutoff_slot"])
            if cutoff != origin - 1:
                raise ValueError(
                    "CAUSAL_CUTOFF_MISMATCH: vehicle prediction must use observations "
                    f"through origin_slot-1; origin={origin}, cutoff={cutoff}"
                )
            vehicle_id = str(row["vehicle_id"])
            if vehicle_id in by_origin.setdefault(origin, {}):
                raise ValueError(
                    "vehicle_predictions.csv duplicate prediction row: "
                    f"origin={origin}, horizon=0, vehicle_id={vehicle_id}"
                )
            order_by_origin.setdefault(origin, []).append(vehicle_id)
            by_origin[origin][vehicle_id] = (
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    if expected_vehicle_ids is not None:
        expected = tuple(str(value) for value in expected_vehicle_ids)
        for origin, ids in order_by_origin.items():
            if len(ids) != len(expected) or set(ids) != set(expected):
                raise ValueError(
                    "vehicle_predictions.csv IDs do not match the observation mapping: "
                    f"origin={origin}, missing={sorted(set(expected) - set(ids))}, "
                    f"unexpected={sorted(set(ids) - set(expected))}"
                )
    return by_origin


def _load_popularity_prediction_h0(
    path: Path,
    allowed_origins: set[int] | None = None,
) -> dict[int, dict[int, tuple[float, int, float]]]:
    by_origin: dict[int, dict[int, tuple[float, int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["horizon"]) != 0:
                continue
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
            cutoff = int(row["observation_cutoff_slot"])
            if cutoff != origin - 1:
                raise ValueError(
                    "CAUSAL_CUTOFF_MISMATCH: popularity prediction must use request "
                    f"history through origin_slot-1; origin={origin}, cutoff={cutoff}"
                )
            service_id = int(row["service_id"])
            by_origin.setdefault(origin, {})[service_id] = (
                float(row["rho_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    return by_origin


def _load_vehicle_prediction_horizon(
    path: Path,
    prediction_horizon: int,
    allowed_origins: set[int] | None = None,
    expected_vehicle_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[int, dict[int, dict[str, tuple[float, float, int, float]]]]:
    """Load only predicted x/y values for every requested prediction horizon."""

    horizon = int(prediction_horizon)
    if horizon <= 0:
        raise ValueError("prediction_horizon 必须大于0。")
    by_origin: dict[int, dict[int, dict[str, tuple[float, float, int, float]]]] = {}
    order_by_origin_horizon: dict[tuple[int, int], list[str]] = {}
    seen: set[tuple[int, int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
            cutoff = int(row["observation_cutoff_slot"])
            if cutoff != origin - 1:
                raise ValueError(
                    "CAUSAL_CUTOFF_MISMATCH: prediction must use observations "
                    f"through origin_slot-1; origin={origin}, cutoff={cutoff}"
                )
            horizon_index = int(row["horizon"])
            if not 0 <= horizon_index < horizon:
                continue
            vehicle_id = str(row["vehicle_id"])
            key = (origin, horizon_index, vehicle_id)
            if key in seen:
                raise ValueError(
                    "vehicle_predictions.csv duplicate prediction row: "
                    f"origin={origin}, horizon={horizon_index}, vehicle_id={vehicle_id}"
                )
            seen.add(key)
            order_by_origin_horizon.setdefault((origin, horizon_index), []).append(
                vehicle_id
            )
            by_origin.setdefault(origin, {}).setdefault(horizon_index, {})[vehicle_id] = (
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    if expected_vehicle_ids is not None:
        expected = tuple(str(value) for value in expected_vehicle_ids)
        for (origin, horizon_index), ids in order_by_origin_horizon.items():
            if len(ids) != len(expected) or set(ids) != set(expected):
                raise ValueError(
                    "vehicle_predictions.csv IDs do not match the observation mapping: "
                    f"origin={origin}, horizon={horizon_index}, "
                    f"missing={sorted(set(expected) - set(ids))}, "
                    f"unexpected={sorted(set(ids) - set(expected))}"
                )
    return by_origin


def _load_popularity_prediction_horizon(
    path: Path,
    prediction_horizon: int,
    allowed_origins: set[int] | None = None,
) -> dict[int, dict[int, dict[int, tuple[float, int, float]]]]:
    """Load rho_pred for every requested prediction horizon; never true labels."""

    horizon = int(prediction_horizon)
    if horizon <= 0:
        raise ValueError("prediction_horizon 必须大于0。")
    by_origin: dict[int, dict[int, dict[int, tuple[float, int, float]]]] = {}
    seen: set[tuple[int, int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
            cutoff = int(row["observation_cutoff_slot"])
            if cutoff != origin - 1:
                raise ValueError(
                    "CAUSAL_CUTOFF_MISMATCH: prediction must use request history "
                    f"through origin_slot-1; origin={origin}, cutoff={cutoff}"
                )
            horizon_index = int(row["horizon"])
            if not 0 <= horizon_index < horizon:
                continue
            service_id = int(row["service_id"])
            key = (origin, horizon_index, service_id)
            if key in seen:
                raise ValueError(
                    "popularity_predictions.csv duplicate prediction row: "
                    f"origin={origin}, horizon={horizon_index}, service_id={service_id}"
                )
            seen.add(key)
            by_origin.setdefault(origin, {}).setdefault(horizon_index, {})[service_id] = (
                float(row["rho_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
            )
    return by_origin


def _load_task_time_by_slot(
    path: Path,
    allowed_slots: set[int] | None = None,
    expected_vehicle_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[int, float]:
    result: dict[int, float] = {}
    vehicle_ids_by_slot: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            slot = int(row["slot"])
            if allowed_slots is not None and slot not in allowed_slots:
                continue
            time_s = float(row["time_s"])
            vehicle_id = str(row["vehicle_id"])
            slot_vehicle_ids = vehicle_ids_by_slot.setdefault(slot, [])
            if vehicle_id in slot_vehicle_ids:
                raise ValueError(
                    f"tasks.csv slot={slot} contains duplicate vehicle {vehicle_id}"
                )
            slot_vehicle_ids.append(vehicle_id)
            if slot in result and not math.isclose(
                result[slot], time_s, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"tasks.csv slot={slot} time_s 不一致。")
            result[slot] = time_s
    if expected_vehicle_ids is not None:
        expected = tuple(str(value) for value in expected_vehicle_ids)
        for slot, ids in vehicle_ids_by_slot.items():
            if len(ids) != len(expected) or set(ids) != set(expected):
                raise ValueError(
                    "tasks.csv vehicle IDs do not match the Observation TV mapping: "
                    f"slot={slot}, missing={sorted(set(expected) - set(ids))}, "
                    f"unexpected={sorted(set(ids) - set(expected))}"
                )
        if allowed_slots is not None:
            missing_slots = set(allowed_slots) - set(vehicle_ids_by_slot)
            if missing_slots:
                raise ValueError(f"tasks.csv is missing selected slots: {sorted(missing_slots)}")
    return result


def prepare_aligned_rollout_steps(
    project_root: Path,
    scenarios: dict[str, Any],
    origin_start_index: int,
    max_steps: int,
    vehicle_prediction_file: Path,
    popularity_prediction_file: Path,
    tasks_file: Path,
    trajectory_file: Path,
    include_bootstrap_step: bool = True,
    *,
    prediction_horizon: int = 1,
    prediction_dt_s: float = 1.0,
    prediction_terminal_origin: int | None = None,
    expected_vehicle_ids: tuple[str, ...] | list[str] | None = None,
    expected_task_vehicle_ids: tuple[str, ...] | list[str] | None = None,
    expected_service_ids: tuple[int, ...] | list[int] | None = None,
    include_prediction_horizon_mask: bool = False,
    arrival_horizon_slots: int | None = None,
    require_full_step_count: bool = False,
) -> list[AlignedRolloutStep]:
    """从正式 artifact 构造已对齐 steps；不重新预测或生成场景。"""

    del project_root  # retained for the stable caller-facing interface
    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0。")
    horizon = int(prediction_horizon)
    if horizon <= 0:
        raise ValueError("prediction_horizon 必须大于0。")
    prediction_dt = float(prediction_dt_s)
    if not math.isfinite(prediction_dt) or prediction_dt <= 0.0:
        raise ValueError("prediction_dt_s 必须是正的有限数。")
    if horizon > 1:
        if prediction_terminal_origin is None:
            raise ValueError("full-horizon alignment requires prediction_terminal_origin")
        if not include_prediction_horizon_mask:
            raise ValueError("full-horizon alignment requires prediction horizon mask")
    elif include_prediction_horizon_mask:
        raise ValueError("prediction horizon mask 只能用于 prediction_horizon > 1")

    n_origin = int(scenarios["origin_slots"].shape[0])
    if origin_start_index < 0 or origin_start_index >= n_origin:
        raise IndexError("origin_start_index 越界。")

    extra_step = 1 if include_bootstrap_step else 0
    requested_stop = origin_start_index + max_steps + extra_step
    if require_full_step_count and requested_stop > n_origin:
        raise ValueError(
            "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
            f"requested origins {origin_start_index}..{requested_stop - 1}, "
            f"available scenario origins 0..{n_origin - 1}"
        )
    stop = min(n_origin, requested_stop)
    selected_origin_indices = list(range(origin_start_index, stop))
    selected_origin_slots = {
        int(cache_configuration.scenario_at_origin(scenarios, index)["origin_slot"])
        for index in selected_origin_indices
    }
    task_slots_for_alignment = selected_origin_slots
    if arrival_horizon_slots is not None:
        if int(arrival_horizon_slots) <= 0:
            raise ValueError("arrival_horizon_slots must be positive")
        task_slots_for_alignment = {
            slot for slot in selected_origin_slots
            if slot < int(arrival_horizon_slots)
        }
    task_time_by_slot = _load_task_time_by_slot(
        tasks_file,
        allowed_slots=task_slots_for_alignment,
        expected_vehicle_ids=expected_task_vehicle_ids,
    )
    if horizon > 1:
        vehicle_pred_horizon = _load_vehicle_prediction_horizon(
            vehicle_prediction_file,
            horizon,
            allowed_origins=selected_origin_slots,
            expected_vehicle_ids=expected_vehicle_ids,
        )
        popularity_pred_horizon = _load_popularity_prediction_horizon(
            popularity_prediction_file,
            horizon,
            allowed_origins=selected_origin_slots,
        )
        vehicle_pred = None
        popularity_pred = None
    else:
        vehicle_pred = _load_vehicle_prediction_h0(
            vehicle_prediction_file,
            allowed_origins=selected_origin_slots,
            expected_vehicle_ids=expected_vehicle_ids,
        )
        popularity_pred = _load_popularity_prediction_h0(
            popularity_prediction_file,
            allowed_origins=selected_origin_slots,
        )
        vehicle_pred_horizon = None
        popularity_pred_horizon = None

    expected_vehicle_ids_t = (
        None
        if expected_vehicle_ids is None
        else tuple(str(value) for value in expected_vehicle_ids)
    )
    expected_task_vehicle_ids_t = (
        None
        if expected_task_vehicle_ids is None
        else tuple(str(value) for value in expected_task_vehicle_ids)
    )
    expected_service_ids_t = (
        None
        if expected_service_ids is None
        else tuple(int(value) for value in expected_service_ids)
    )
    steps: list[AlignedRolloutStep] = []
    for origin_index in selected_origin_indices:
        scenario = cache_configuration.scenario_at_origin(scenarios, origin_index)
        slot = int(scenario["origin_slot"])
        if require_full_step_count and slot != origin_index:
            raise ValueError(
                "complete lifecycle origins must be contiguous zero-based slots: "
                f"origin_index={origin_index}, scenario_slot={slot}"
            )
        time_s = float(scenario["origin_time_s"])
        task_time = task_time_by_slot.get(slot)
        if task_time is None:
            if arrival_horizon_slots is None or slot < int(arrival_horizon_slots):
                raise ValueError(
                    f"origin_index={origin_index} 的正式到达 slot 缺少 tasks.csv 时间记录。"
                )
        elif not math.isclose(
            float(task_time), time_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"origin_index={origin_index} 的 scenario slot/time 与 tasks.csv 不一致。")
        if int(scenario["horizons"][0]) != 0:
            raise ValueError("scenario horizons[0] 必须为0，表示当前调度slot。")

        vehicle_ids = tuple(str(value) for value in scenario["vehicle_ids"].tolist())
        task_vehicle_ids = tuple(
            str(value) for value in scenario["task_vehicle_ids"].tolist()
        )
        if (
            not task_vehicle_ids
            or len(set(task_vehicle_ids)) != len(task_vehicle_ids)
            or not set(task_vehicle_ids).issubset(vehicle_ids)
        ):
            raise ValueError("scenario task_vehicle_ids must be unique members of vehicle_ids")
        if expected_task_vehicle_ids_t is not None and set(task_vehicle_ids) != set(
            expected_task_vehicle_ids_t
        ):
            raise ValueError(
                "scenario task vehicle IDs differ from the fixed Observation order: "
                f"origin={origin_index}"
            )
        if (
            expected_task_vehicle_ids_t is not None
            and task_vehicle_ids != expected_task_vehicle_ids_t
        ):
            raise ValueError(
                "scenario task vehicle ID order differs from ObservationSpec: "
                f"origin={origin_index}"
            )
        service_vehicle_ids = tuple(
            str(value) for value in scenario["service_vehicle_ids"].tolist()
        )
        vehicle_roles = tuple(str(value) for value in scenario["vehicle_roles"].tolist())
        expected_roles = ("SV",) * len(service_vehicle_ids) + (
            "TV",
        ) * len(task_vehicle_ids)
        if (
            vehicle_roles != expected_roles
            or vehicle_ids != service_vehicle_ids + task_vehicle_ids
        ):
            raise ValueError(
                "scenario vehicle IDs/roles do not match the SV-then-TV mapping: "
                f"origin={origin_index}"
            )
        if expected_vehicle_ids_t is not None:
            if vehicle_ids != expected_vehicle_ids_t:
                raise ValueError(
                    "scenario vehicle IDs/order 与 observation spec 不一致："
                    f"origin={origin_index}"
                )
        observation_vehicle_ids = expected_vehicle_ids_t or vehicle_ids
        if set(observation_vehicle_ids) != set(vehicle_ids):
            raise ValueError("observation vehicle IDs 与 scenario vehicle IDs 不一致。")

        service_ids = tuple(range(int(scenario["rho_hat"].shape[1])))
        if horizon > 1 and expected_service_ids_t is not None:
            if service_ids != expected_service_ids_t:
                raise ValueError(
                    "scenario service IDs/order 与 observation spec 不一致："
                    f"origin={origin_index}"
                )
        observation_service_ids = expected_service_ids_t or service_ids
        if observation_service_ids != tuple(range(len(observation_service_ids))):
            raise ValueError("observation service IDs 必须严格为 0..F-1。")

        if horizon > 1:
            assert vehicle_pred_horizon is not None
            assert popularity_pred_horizon is not None
            valid_mask = prediction_horizon_valid_mask(
                origin_index,
                horizon,
                int(prediction_terminal_origin),
            )
            predicted_vehicle_horizon: list[dict[str, np.ndarray]] = []
            predicted_popularity_horizon: list[dict[int, float]] = []
            origin_vehicle_rows = vehicle_pred_horizon.get(slot, {})
            origin_popularity_rows = popularity_pred_horizon.get(slot, {})
            for horizon_index in range(horizon):
                if not bool(valid_mask[horizon_index]):
                    predicted_vehicle_horizon.append({})
                    predicted_popularity_horizon.append({})
                    continue
                expected_slot = slot + horizon_index
                expected_time = time_s + horizon_index * prediction_dt
                raw_vehicle = origin_vehicle_rows.get(horizon_index)
                if raw_vehicle is None or set(raw_vehicle) != set(vehicle_ids):
                    if (
                        arrival_horizon_slots is not None
                        and slot >= int(arrival_horizon_slots)
                    ):
                        raise ValueError(
                            "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
                            "vehicle_predictions.csv lacks decision-time forecasts "
                            f"for execution slot={slot}, horizon={horizon_index}; "
                            "future trajectory labels are not valid substitutes."
                        )
                    raise ValueError(
                        "vehicle_predictions.csv 缺少完整 horizon 预测："
                        f"origin slot={slot}, horizon={horizon_index}"
                    )
                positions: dict[str, np.ndarray] = {}
                for vehicle_id in observation_vehicle_ids:
                    x_pred, y_pred, target_slot, target_time = raw_vehicle[vehicle_id]
                    if target_slot != expected_slot or not math.isclose(
                        target_time,
                        expected_time,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "vehicle prediction target 与 slot/time 不一致："
                            f"slot={slot}, horizon={horizon_index}"
                        )
                    if not math.isfinite(x_pred) or not math.isfinite(y_pred):
                        raise ValueError("vehicle prediction x_pred/y_pred 必须有限。")
                    positions[vehicle_id] = np.asarray([x_pred, y_pred], dtype=np.float64)
                predicted_vehicle_horizon.append(positions)

                raw_popularity = origin_popularity_rows.get(horizon_index)
                if raw_popularity is None or set(raw_popularity) != set(service_ids):
                    if (
                        arrival_horizon_slots is not None
                        and slot >= int(arrival_horizon_slots)
                    ):
                        raise ValueError(
                            "MISSING_CAUSAL_EXECUTION_TAIL_INPUTS: "
                            "popularity_predictions.csv lacks decision-time forecasts "
                            f"for execution slot={slot}, horizon={horizon_index}."
                        )
                    raise ValueError(
                        "popularity_predictions.csv 缺少完整 horizon 预测："
                        f"origin slot={slot}, horizon={horizon_index}"
                    )
                popularity: dict[int, float] = {}
                for service_id in observation_service_ids:
                    rho, target_slot, target_time = raw_popularity[service_id]
                    if target_slot != expected_slot or not math.isclose(
                        target_time,
                        expected_time,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "popularity prediction target 与 slot/time 不一致："
                            f"slot={slot}, horizon={horizon_index}"
                        )
                    if not math.isfinite(rho) or rho < -1e-8:
                        raise ValueError("popularity rho_pred 必须是有限非负值。")
                    popularity[service_id] = rho
                if not math.isclose(
                    sum(popularity.values()),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        f"slot={slot}, horizon={horizon_index} popularity rho_pred 未归一化。"
                    )
                predicted_popularity_horizon.append(popularity)
            predicted_vehicle_horizon_t = tuple(predicted_vehicle_horizon)
            predicted_popularity_horizon_t = tuple(predicted_popularity_horizon)
            predicted_positions = predicted_vehicle_horizon_t[0]
            predicted_popularity = predicted_popularity_horizon_t[0]
            prediction_horizon_mask: np.ndarray | None = valid_mask
        else:
            assert vehicle_pred is not None
            assert popularity_pred is not None
            raw_vehicle = vehicle_pred.get(slot)
            if raw_vehicle is None or set(raw_vehicle) != set(vehicle_ids):
                raise ValueError(f"vehicle_predictions.csv 缺少 origin slot={slot} 的完整车辆预测。")
            predicted_positions = {}
            for vehicle_id in observation_vehicle_ids:
                x_pred, y_pred, target_slot, target_time = raw_vehicle[vehicle_id]
                if target_slot != slot or not math.isclose(
                    target_time,
                    time_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"vehicle prediction target 与 slot/time 不一致：slot={slot}")
                predicted_positions[vehicle_id] = np.asarray([x_pred, y_pred], dtype=np.float64)
            raw_popularity = popularity_pred.get(slot)
            if raw_popularity is None or set(raw_popularity) != set(service_ids):
                raise ValueError(f"popularity_predictions.csv 缺少 origin slot={slot} 的完整服务预测。")
            predicted_popularity = {}
            for service_id in service_ids:
                rho, target_slot, target_time = raw_popularity[service_id]
                if target_slot != slot or not math.isclose(
                    target_time,
                    time_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(f"popularity prediction target 与 slot/time 不一致：slot={slot}")
                predicted_popularity[service_id] = rho
            if not math.isclose(
                sum(predicted_popularity.values()),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(f"slot={slot} popularity rho_pred 未归一化。")
            predicted_vehicle_horizon_t = (predicted_positions,)
            predicted_popularity_horizon_t = (predicted_popularity,)
            prediction_horizon_mask = None

        actual_tasks = task_allocation.load_actual_tasks(tasks_file, slot)
        if (
            arrival_horizon_slots is not None
            and slot >= int(arrival_horizon_slots)
            and actual_tasks
        ):
            raise ValueError(
                "new task arrivals are forbidden after arrival_horizon_slots: "
                f"slot={slot}, T={int(arrival_horizon_slots)}, arrivals={len(actual_tasks)}"
            )
        if expected_task_vehicle_ids_t is not None:
            unexpected_task_vehicles = {
                str(task.vehicle_id) for task in actual_tasks
            } - set(expected_task_vehicle_ids_t)
            if unexpected_task_vehicles:
                raise ValueError(
                    "tasks.csv contains a task vehicle outside the Observation TV order: "
                    f"slot={slot}, vehicles={sorted(unexpected_task_vehicles)}"
                )
        sv_ids = {str(value) for value in scenario["service_vehicle_ids"].tolist()}
        # Load positions for current-slot arrivals and all TVs present in this
        # trajectory slice; the formal per-slot execution path has no historical
        # task work that would require departed-TV positions.
        tv_ids = {str(value) for value in scenario["vehicle_ids"].tolist()}
        actual_positions = task_allocation.load_vehicle_positions(
            trajectory_file,
            time_s,
            sv_ids | tv_ids,
            require_all=False,
        )
        required_current = sv_ids | {task.vehicle_id for task in actual_tasks}
        missing_required = required_current - set(actual_positions)
        if missing_required:
            if (
                arrival_horizon_slots is not None
                and slot >= int(arrival_horizon_slots)
            ):
                raise ValueError(
                    "UNRESOLVED_VEHICLE_DEPARTURE_SEMANTICS: "
                    f"trajectory.csv lacks required current physical positions at "
                    f"execution slot={slot}, time_s={time_s}: "
                    f"{sorted(missing_required)}; no hold, extrapolation, migration, "
                    "automatic failure, or fill policy is configured."
                )
            raise ValueError(
                f"trajectory.csv 缺少当前 time_s={time_s} 的服务车辆/到达任务位置："
                f"{sorted(missing_required)}"
            )
        steps.append(
            AlignedRolloutStep(
                origin_index=origin_index,
                slot=slot,
                time_s=time_s,
                scenario_slice=scenario,
                predicted_vehicle_positions=predicted_positions,
                predicted_service_popularity=predicted_popularity,
                predicted_vehicle_positions_horizon=predicted_vehicle_horizon_t,
                predicted_service_popularity_horizon=predicted_popularity_horizon_t,
                prediction_horizon_mask=prediction_horizon_mask,
                actual_tasks=actual_tasks,
                actual_vehicle_positions=actual_positions,
            )
        )
    for left, right in zip(steps, steps[1:]):
        if right.origin_index <= left.origin_index or right.slot <= left.slot or right.time_s <= left.time_s:
            raise ValueError("aligned steps 的 origin/slot/time 必须严格递增。")
    return steps
