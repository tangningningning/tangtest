"""定义正式环境的状态、动作和 artifact 对齐接口。"""

from __future__ import annotations

import csv
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


PROPOSED_POLICY_SCHEMA = project_config.PROPOSED_POLICY_SCHEMA
MULTI_NODE_FACTORIZED_POLICY_SCHEMA = project_config.MULTI_NODE_FACTORIZED_POLICY_SCHEMA
SPARSE_COORDINATED_CACHE_POLICY_SCHEMA = project_config.SPARSE_COORDINATED_CACHE_POLICY_SCHEMA
PROPOSED_CONTROL_MODE = "macro_cache_smdp_micro_uav_candidate"
PER_SLOT_ANALYTIC_TASK_MODEL = "PER_SLOT_ANALYTIC"
PER_SLOT_ANALYTIC_OBSERVATION_SCHEMA = "configuration_per_slot_analytic_v1"


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
    terminal_origin: int
    configured_rollout_steps: int | None
    minibatch_size: int
    configuration_control_mode: str
    policy_schema: str
    cache_action_mode: str
    cache_reconfiguration_model: str
    realization: "RealizationArtifacts | None" = None
    observation_time_to_go: Callable[[int], float] | None = None

def _assert_realization_context_compatible(
    reference: EnvironmentContext,
    candidate: EnvironmentContext,
) -> None:
    """确认新 realization 满足当前 PPO 张量与协议约定。"""

    scalar_fields = (
        "data_protocol",
        "episode_slots",
        "train_origin_start",
        "train_origin_end",
        "terminal_origin",
        "configured_rollout_steps",
        "minibatch_size",
        "configuration_control_mode",
        "policy_schema",
        "cache_action_mode",
        "cache_reconfiguration_model",
    )
    for field in scalar_fields:
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(
                "PPO episode realization changes the fixed training contract: "
                f"field={field}, reference={getattr(reference, field)!r}, "
                f"candidate={getattr(candidate, field)!r}"
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
            "PPO episode realization changes the observation tensor contract: "
            f"reference_dim={reference_observation.observation_dim}, "
            f"candidate_dim={candidate_observation.observation_dim}"
        )
    reference_cache_roles = tuple(
        str(value).upper() for value in reference.cache_nodes["node_roles"]
    )
    candidate_cache_roles = tuple(
        str(value).upper() for value in candidate.cache_nodes["node_roles"]
    )
    # Service-vehicle IDs may differ across full_episode_randomized realizations.
    # Each episode builds and saves observations/candidate descriptors in its own
    # action order, so the cross-episode tensor contract is the role ordering.
    if (
        len(reference.action_spec.cache_node_ids)
        != len(candidate.action_spec.cache_node_ids)
        or reference_cache_roles != candidate_cache_roles
        or reference.action_spec.uav_ids != candidate.action_spec.uav_ids
    ):
        raise ValueError("PPO realization changes Cache node role ordering or UAV ID ordering.")
    if np.asarray(reference.initial_cache).shape != np.asarray(candidate.initial_cache).shape:
        raise ValueError("PPO episode realization changes the initial cache tensor shape")
    if set(reference.initial_uav) != set(candidate.initial_uav):
        raise ValueError("PPO episode realization changes the initial UAV ID set")

def load_environment_context(
    config_path: str = "configs/proposed/system_config.toml",
    realization_dir: str | Path | None = None,
) -> EnvironmentContext:
    cfg = project_config.load_config(config_path)
    project_config.validate_for_formal_experiment(cfg)
    data_protocol = project_config.get_ppo_data_protocol(cfg)
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
    action_spec = ConfigurationActionSpec.from_cache_nodes_and_project(cache_nodes, cfg)
    policy_schema = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.policy_schema",
            PROPOSED_POLICY_SCHEMA,
        )
    )
    cache_action_mode = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.cache_action_mode",
            project_config.CACHE_ACTION_MODE_SINGLE_NODE,
        )
    )
    cache_reconfiguration_model = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.cache_reconfiguration_model",
            "SINGLE_NODE_MACRO_BOUNDARY_RECONFIGURATION",
        )
    )
    expected_schema = {
        project_config.CACHE_ACTION_MODE_SINGLE_NODE: PROPOSED_POLICY_SCHEMA,
        project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED: MULTI_NODE_FACTORIZED_POLICY_SCHEMA,
        project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED: SPARSE_COORDINATED_CACHE_POLICY_SCHEMA,
    }.get(cache_action_mode)
    if expected_schema is None:
        raise ValueError(f"unsupported Cache action mode: {cache_action_mode}")
    if policy_schema != expected_schema:
        raise ValueError(
            "Cache action mode 与 policy schema 不匹配："
            f"mode={cache_action_mode}, expected_schema={expected_schema}, "
            f"actual_schema={policy_schema}"
        )
    if cache_action_mode not in {
        project_config.CACHE_ACTION_MODE_SINGLE_NODE,
        project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED,
        project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED,
    }:
        raise ValueError(f"unsupported Cache action mode: {cache_action_mode}")
    if cache_action_mode in {
        project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED,
        project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED,
    }:
        if cache_reconfiguration_model != project_config.CACHE_RECONFIGURATION_MODEL_ABSTRACT:
            raise ValueError("multi-node mode requires the explicit abstract reconfiguration model")
        roles = [str(value).upper() for value in cache_nodes["node_roles"]]
        if (
            not action_spec.cache_node_ids
            or len(set(action_spec.cache_node_ids)) != len(action_spec.cache_node_ids)
            or len(roles) != len(action_spec.cache_node_ids)
            or any(role not in {"SV", "RSU", "UAV"} for role in roles)
            or not action_spec.uav_ids
        ):
            raise ValueError(
                "Multi-node MAPPO requires unique ordered Cache nodes with supported roles and at least one UAV."
            )
        if cache_action_mode == project_config.CACHE_ACTION_MODE_SPARSE_COORDINATED:
            max_updates = project_config.get_config_value(
                cfg, "algorithm.ppo.max_cache_updates_per_frame", None
            )
            if (
                isinstance(max_updates, bool)
                or not isinstance(max_updates, int)
                or max_updates != 1
            ):
                raise ValueError(
                    "sparse coordinated mode requires max_cache_updates_per_frame=1"
                )
    elif cache_reconfiguration_model != "SINGLE_NODE_MACRO_BOUNDARY_RECONFIGURATION":
        raise ValueError("single-node mode has an unsupported cache reconfiguration model")
    ppo_config = cfg["algorithm"]["ppo"]
    if ppo_config.get("task_execution_model") != PER_SLOT_ANALYTIC_TASK_MODEL:
        raise ValueError(
            "algorithm.ppo.task_execution_model must be PER_SLOT_ANALYTIC"
        )
    required_flags = {
        "macro_cache_enabled": True,
        "macro_full_batch": True,
        "exact_q_training_enabled": False,
        "critic_mc_warmup_required": False,
        "uav_unified_candidate_policy": True,
        "uav_autoregressive": True,
        "uav_group_balance": True,
    }
    for key, expected in required_flags.items():
        if ppo_config.get(key) is not expected:
            raise ValueError(f"algorithm.ppo.{key} 必须为 {expected!r}。")
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
    configuration_control_mode = str(
        project_config.get_config_value(
            cfg,
            "algorithm.ppo.configuration_control_mode",
            PROPOSED_CONTROL_MODE,
        )
    )
    if configuration_control_mode != PROPOSED_CONTROL_MODE:
        raise ValueError(
            "当前训练器仅支持正式 proposed_joint 控制模式："
            f"{PROPOSED_CONTROL_MODE}"
        )
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
        terminal_origin=int(data_protocol["terminal_origin"]),
        configured_rollout_steps=(
            None
            if data_protocol["rollout_steps"] is None
            else int(data_protocol["rollout_steps"])
        ),
        minibatch_size=int(data_protocol["minibatch_size"]),
        configuration_control_mode=configuration_control_mode,
        policy_schema=policy_schema,
        cache_action_mode=cache_action_mode,
        cache_reconfiguration_model=cache_reconfiguration_model,
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
    """PPO configuration observation 的固定结构。"""

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
    edge_service_lock_slice: slice | None
    edge_upload_workload_slice: slice | None
    edge_compute_workload_slice: slice | None
    cloud_backlog_slice: slice | None
    uav_tv_upload_slice: slice | None
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
        include_ledger_features: bool = False,
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
        if include_ledger_features:
            edge_service_feature_len = len(cache_node_ids_t) * len(service_ids_t)
            edge_service_lock_slice = slice(start, start + edge_service_feature_len)
            start += edge_service_feature_len
            edge_upload_workload_slice = slice(start, start + edge_service_feature_len)
            start += edge_service_feature_len
            edge_compute_workload_slice = slice(start, start + edge_service_feature_len)
            start += edge_service_feature_len
            cloud_backlog_slice = slice(start, start + 4)
            start += 4
            uav_tv_upload_slice = slice(
                start, start + len(uav_ids_t) * len(task_vehicle_ids_t)
            )
            start += len(uav_ids_t) * len(task_vehicle_ids_t)
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
        include_ledger_features: bool = False,
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
            include_ledger_features=include_ledger_features,
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


LEDGER_FEATURE_SCHEMA = "configuration_ledger_state_v1"
LEDGER_FEATURE_NORMALIZATION = "bounded_log1p_ratio_v1"
LEDGER_FEATURE_SLOT_SECONDS = 1.0


def _normalize_ledger_workload(values: Any, references: Any, label: str) -> np.ndarray:
    """Map finite nonnegative physical work to a bounded monotone [0, 1) value."""

    workload = np.asarray(values, dtype=np.float64)
    reference = np.asarray(references, dtype=np.float64)
    if not np.all(np.isfinite(workload)) or np.any(workload < 0.0):
        raise ValueError(f"{label} must be finite and nonnegative")
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError(f"{label} normalization reference must be finite and positive")
    try:
        workload, reference = np.broadcast_arrays(workload, reference)
    except ValueError as exc:
        raise ValueError(
            f"{label} shape is incompatible with its normalization reference"
        ) from exc
    log_load = np.zeros(workload.shape, dtype=np.float64)
    positive = workload > 0.0
    log_ratio = np.log(workload[positive]) - np.log(reference[positive])
    log_load[positive] = np.logaddexp(0.0, log_ratio)
    normalized = log_load / (1.0 + log_load)
    if (
        not np.all(np.isfinite(normalized))
        or np.any(normalized < 0.0)
        or np.any(normalized >= 1.0)
    ):
        raise FloatingPointError(f"{label} normalization produced an invalid value")
    return normalized


def _ledger_finite_nonnegative(value: Any, field_name: str, task_id: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_name}[{task_id!r}] must be finite and nonnegative")
    return numeric


def _ledger_observation_features(
    cache_state: Any,
    task_ledger: slot_environment.CrossSlotTaskLedger,
    observation_spec: ConfigurationObservationSpec,
    service_catalog: dict[str, Any],
    cache_nodes: dict[str, Any],
    cache_cfg: cache_configuration.CacheOptimizationConfig,
) -> np.ndarray:
    """Build the appended workload features from the current persistent task ledger."""

    ledger_slices = (
        observation_spec.edge_service_lock_slice,
        observation_spec.edge_upload_workload_slice,
        observation_spec.edge_compute_workload_slice,
        observation_spec.cloud_backlog_slice,
        observation_spec.uav_tv_upload_slice,
    )
    if any(value is None for value in ledger_slices):
        raise ValueError("Ledger features require an observation spec that includes them")
    node_ids = tuple(str(value) for value in cache_nodes["node_ids"])
    node_roles = tuple(str(value).upper() for value in cache_nodes["node_roles"])
    if node_ids != observation_spec.cache_node_ids or len(node_roles) != len(node_ids):
        raise ValueError(
            "Ledger Cache node order differs from the observation specification"
        )
    catalog_service_ids = tuple(
        int(value) for value in np.asarray(service_catalog["service_ids"]).tolist()
    )
    if catalog_service_ids != observation_spec.service_ids:
        raise ValueError("Ledger service order differs from the observation specification")

    locks = slot_environment.active_cache_service_locks(
        task_ledger,
        np.asarray(cache_state, dtype=np.int8),
        service_catalog,
        cache_nodes,
        expected_node_ids=observation_spec.cache_node_ids,
        expected_service_ids=observation_spec.service_ids,
    )
    node_index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    service_index_by_id = {
        service_id: index for index, service_id in enumerate(observation_spec.service_ids)
    }
    uav_index_by_id = {
        uav_id: index for index, uav_id in enumerate(observation_spec.uav_ids)
    }
    task_vehicle_index_by_id = {
        vehicle_id: index
        for index, vehicle_id in enumerate(observation_spec.task_vehicle_ids)
    }
    role_cpu_rates = {
        "SV": float(cache_cfg.sv_cpu_cycles_per_s),
        "RSU": float(cache_cfg.rsu_cpu_cycles_per_s),
        "UAV": float(cache_cfg.uav_cpu_cycles_per_s),
    }
    unknown_roles = sorted(set(node_roles) - set(role_cpu_rates))
    if unknown_roles:
        raise ValueError(
            f"unsupported Edge Cache node roles in Ledger observation: {unknown_roles}"
        )
    edge_cpu_references = np.asarray(
        [role_cpu_rates[role] * LEDGER_FEATURE_SLOT_SECONDS for role in node_roles],
        dtype=np.float64,
    )[:, None]
    upload_reference_bits = (
        float(cache_cfg.cloud_uplink_rate_bps) * LEDGER_FEATURE_SLOT_SECONDS
    )
    cloud_cpu_reference = (
        float(cache_cfg.cloud_cpu_capacity_cycles_per_s)
        * LEDGER_FEATURE_SLOT_SECONDS
    )
    if not np.isfinite(upload_reference_bits) or upload_reference_bits <= 0.0:
        raise ValueError("Cloud uplink rate must be positive to normalize upload workload")
    if not np.isfinite(cloud_cpu_reference) or cloud_cpu_reference <= 0.0:
        raise ValueError("Cloud CPU capacity must be positive to normalize compute workload")

    edge_upload_bits = np.zeros((len(node_ids), len(service_index_by_id)), dtype=np.float64)
    edge_cpu_cycles = np.zeros_like(edge_upload_bits)
    cloud_task_count = 0.0
    cloud_upload_bits = 0.0
    cloud_cpu_cycles = 0.0
    cloud_extra_delay_s = 0.0
    uav_tv_upload_bits = np.zeros(
        (len(observation_spec.uav_ids), len(observation_spec.task_vehicle_ids)),
        dtype=np.float64,
    )
    allowed_stages = {"UPLOADING", "COMPUTING", "NETWORK_DELAY"}
    for task_id, state in sorted(task_ledger.active.items(), key=lambda item: str(item[0])):
        if str(task_id) != str(state.task_id):
            raise ValueError(f"active task ledger key mismatch: {task_id}")
        stage = str(state.execution_stage).upper()
        if stage not in allowed_stages:
            raise ValueError(
                f"task {task_id} has invalid active execution stage at observation boundary: {stage}"
            )
        service_id = int(state.service_id)
        if service_id not in service_index_by_id:
            raise ValueError(f"task {task_id} references unknown service ID {service_id}")
        vehicle_id = str(state.vehicle_id)
        if vehicle_id not in task_vehicle_index_by_id:
            raise ValueError(
                f"task {task_id} vehicle {vehicle_id!r} is absent from formal vehicle order"
            )
        remaining_input_bits = _ledger_finite_nonnegative(
            state.remaining_input_bits, "remaining_input_bits", task_id
        )
        remaining_cpu_cycles = _ledger_finite_nonnegative(
            state.remaining_cpu_cycles, "remaining_cpu_cycles", task_id
        )
        remaining_network_delay_s = _ledger_finite_nonnegative(
            state.remaining_network_delay_s, "remaining_network_delay_s", task_id
        )
        assigned_node = str(state.assigned_node)
        if assigned_node == task_allocation.CLOUD:
            cloud_task_count += 1.0
            if stage == "UPLOADING":
                cloud_upload_bits += remaining_input_bits
            cloud_cpu_cycles += remaining_cpu_cycles
            cloud_extra_delay_s += remaining_network_delay_s
            continue
        if assigned_node not in node_index_by_id:
            raise ValueError(
                f"task {task_id} references unknown Edge node {assigned_node!r}"
            )
        node_index = node_index_by_id[assigned_node]
        service_index = service_index_by_id[service_id]
        edge_cpu_cycles[node_index, service_index] += remaining_cpu_cycles
        if stage == "UPLOADING":
            edge_upload_bits[node_index, service_index] += remaining_input_bits
            if assigned_node in uav_index_by_id:
                uav_tv_upload_bits[
                    uav_index_by_id[assigned_node], task_vehicle_index_by_id[vehicle_id]
                ] += remaining_input_bits

    normalized_edge_upload = _normalize_ledger_workload(
        edge_upload_bits,
        upload_reference_bits,
        "Edge remaining upload bits",
    )
    normalized_edge_cpu = _normalize_ledger_workload(
        edge_cpu_cycles,
        edge_cpu_references,
        "Edge remaining CPU cycles",
    )
    normalized_cloud_backlog = np.asarray(
        [
            float(
                _normalize_ledger_workload(
                    cloud_task_count, 1.0, "Cloud active task count"
                )
            ),
            float(
                _normalize_ledger_workload(
                    cloud_upload_bits,
                    upload_reference_bits,
                    "Cloud remaining upload bits",
                )
            ),
            float(
                _normalize_ledger_workload(
                    cloud_cpu_cycles,
                    cloud_cpu_reference,
                    "Cloud remaining CPU cycles",
                )
            ),
            float(
                _normalize_ledger_workload(
                    cloud_extra_delay_s,
                    LEDGER_FEATURE_SLOT_SECONDS,
                    "Cloud extra network delay",
                )
            ),
        ],
        dtype=np.float64,
    )
    normalized_uav_tv_upload = _normalize_ledger_workload(
        uav_tv_upload_bits,
        upload_reference_bits,
        "UAV/TV remaining upload bits",
    )
    features = np.concatenate(
        [
            locks.reshape(-1).astype(np.float64),
            normalized_edge_upload.reshape(-1),
            normalized_edge_cpu.reshape(-1),
            normalized_cloud_backlog,
            normalized_uav_tv_upload.reshape(-1),
        ]
    )
    expected_size = (
        sum(int(value.stop) - int(value.start) for value in ledger_slices)
    )
    if features.shape != (expected_size,) or not np.all(np.isfinite(features)):
        raise ValueError("Ledger observation features have invalid shape or non-finite values")
    if np.any(features < 0.0) or np.any(features > 1.0):
        raise ValueError("Ledger observation features must lie in [0, 1]")
    return features


def build_configuration_observation(
    previous_cache_state: Any,
    previous_uav_positions: dict[str, Any] | Any,
    predicted_vehicle_positions: dict[str, Any] | Any,
    predicted_service_popularity: dict[int | str, float] | Any,
    observation_spec: ConfigurationObservationSpec,
    flight_boundary: tuple[float, float, float, float],
    *,
    task_ledger: slot_environment.CrossSlotTaskLedger | None = None,
    service_catalog: dict[str, Any] | None = None,
    cache_nodes: dict[str, Any] | None = None,
    cache_cfg: cache_configuration.CacheOptimizationConfig | None = None,
    time_to_go: float | None = None,
    prediction_horizon_mask: Any = None,
) -> np.ndarray:
    """Build the formal state, optionally appending legacy Ledger features."""

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

    if task_ledger is None:
        if any(
            value is not None
            for value in (service_catalog, cache_nodes, cache_cfg)
        ):
            raise ValueError(
                "Ledger observation dependencies must be omitted when no Ledger is supplied"
            )
        ledger_features = np.empty(0, dtype=np.float64)
    else:
        if service_catalog is None or cache_nodes is None or cache_cfg is None:
            raise ValueError("Ledger observation requires its catalog, nodes, and cache config")
        ledger_features = _ledger_observation_features(
            previous_cache,
            task_ledger,
            observation_spec,
            service_catalog,
            cache_nodes,
            cache_cfg,
        )
    observation = np.concatenate(
        [
            cache_features,
            uav_features,
            vehicle_features,
            popularity_features,
            horizon_mask_features,
            time_features,
            ledger_features,
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
    task_ledger: slot_environment.CrossSlotTaskLedger | None = None,
    service_catalog: dict[str, Any] | None = None,
    cache_nodes: dict[str, Any] | None = None,
    cache_cfg: cache_configuration.CacheOptimizationConfig | None = None,
    time_to_go_override: float | None = None,
) -> np.ndarray:
    """构造 Macro Cache 与 Micro UAV 共用的正式 slot observation。"""

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
        task_ledger=task_ledger,
        service_catalog=service_catalog,
        cache_nodes=cache_nodes,
        cache_cfg=cache_cfg,
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
) -> dict[int, dict[str, tuple[float, float, int, float]]]:
    by_origin: dict[int, dict[str, tuple[float, float, int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row["horizon"]) != 0:
                continue
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
            by_origin.setdefault(origin, {})[str(row["vehicle_id"])] = (
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
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
) -> dict[int, dict[int, dict[str, tuple[float, float, int, float]]]]:
    """Load only predicted x/y values for every requested prediction horizon."""

    horizon = int(prediction_horizon)
    if horizon <= 0:
        raise ValueError("prediction_horizon 必须大于0。")
    by_origin: dict[int, dict[int, dict[str, tuple[float, float, int, float]]]] = {}
    seen: set[tuple[int, int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            origin = int(row["origin_slot"])
            if allowed_origins is not None and origin not in allowed_origins:
                continue
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
            by_origin.setdefault(origin, {}).setdefault(horizon_index, {})[vehicle_id] = (
                float(row["x_pred"]),
                float(row["y_pred"]),
                int(row["target_slot"]),
                float(row["target_time_s"]),
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
) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            slot = int(row["slot"])
            if allowed_slots is not None and slot not in allowed_slots:
                continue
            time_s = float(row["time_s"])
            if slot in result and not math.isclose(
                result[slot], time_s, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"tasks.csv slot={slot} time_s 不一致。")
            result[slot] = time_s
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
    stop = min(n_origin, origin_start_index + max_steps + extra_step)
    selected_origin_indices = list(range(origin_start_index, stop))
    selected_origin_slots = {
        int(cache_configuration.scenario_at_origin(scenarios, index)["origin_slot"])
        for index in selected_origin_indices
    }
    task_time_by_slot = _load_task_time_by_slot(
        tasks_file,
        allowed_slots=selected_origin_slots,
    )
    if horizon > 1:
        vehicle_pred_horizon = _load_vehicle_prediction_horizon(
            vehicle_prediction_file,
            horizon,
            allowed_origins=selected_origin_slots,
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
        time_s = float(scenario["origin_time_s"])
        if not math.isclose(
            float(task_time_by_slot.get(slot, float("nan"))),
            time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
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
        if horizon > 1 and expected_vehicle_ids_t is not None:
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
            for vehicle_id in vehicle_ids:
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
