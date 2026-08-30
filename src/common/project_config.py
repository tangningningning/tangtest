#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一项目配置读取、override、校验和稳定 hash。

本文件只处理配置，不放业务算法逻辑。Python 3.11+ 使用标准库 tomllib；
若运行环境低于 3.11，则尝试使用轻量 tomli。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def find_project_root() -> Path:
    candidates = (Path(__file__).resolve().parents[2], Path.cwd().resolve())
    for candidate in candidates:
        if (candidate / "system_config.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("无法找到包含 system_config.toml 的项目根目录。")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_config(path: str | Path = "system_config.toml") -> dict[str, Any]:
    project_root = find_project_root()
    config_path = resolve_path(project_root, path)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if not isinstance(config, dict):
        raise ValueError("system_config.toml 根对象必须是 table。")
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    validate_config(config)
    return config


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if not overrides:
        validate_config(updated)
        return updated
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set override 必须形如 a.b=value：{override}")
        dotted_key, raw_value = override.split("=", 1)
        parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not parts:
            raise ValueError(f"override key 为空：{override}")
        current: Any = updated
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise KeyError(f"override 路径不存在或不是 table：{'.'.join(parts[:-1])}")
            current = current[part]
        leaf = parts[-1]
        if leaf not in current:
            raise KeyError(f"override key 不存在：{dotted_key}")
        old_value = current[leaf]
        new_value = parse_override_value(raw_value)
        if old_value is not None and not isinstance(new_value, type(old_value)):
            if isinstance(old_value, float) and isinstance(new_value, int):
                new_value = float(new_value)
            elif isinstance(old_value, int) and isinstance(new_value, float) and new_value.is_integer():
                new_value = int(new_value)
            else:
                raise TypeError(
                    f"override {dotted_key} 类型不匹配：原类型 {type(old_value).__name__}，"
                    f"新类型 {type(new_value).__name__}"
                )
        current[leaf] = new_value
    validate_config(updated)
    return updated


def config_hash(config: dict[str, Any]) -> str:
    canonical = copy.deepcopy(config)
    canonical.pop("_meta", None)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_parameter_status(config: dict[str, Any], dotted_key: str) -> str | None:
    statuses = config.get("parameter_status", {})
    value = statuses.get(dotted_key)
    return str(value) if value is not None else None


def list_unresolved_parameters(config: dict[str, Any]) -> dict[str, str]:
    statuses = config.get("parameter_status", {})
    allowed = {
        "CONFIRMED_DATA",
        "PROVISIONAL_IMPLEMENTATION",
        "TODO_LITERATURE_CALIBRATION",
    }
    unresolved: dict[str, str] = {}
    for key, value in statuses.items():
        status = str(value)
        if status not in allowed:
            raise ValueError(f"未知参数状态 {status}：{key}")
        if status in {"PROVISIONAL_IMPLEMENTATION", "TODO_LITERATURE_CALIBRATION"}:
            unresolved[str(key)] = status
    return unresolved


# 只把真正影响系统物理模型、约束和正式目标函数的参数纳入 formal status gate。
# 算法超参数（如 local search 轮数/容差、surrogate range cutoff）仍由 validate_config()
# 检查类型和取值范围，但不要求伪装成“文献物理校准参数”。
FORMAL_SYSTEM_PARAMETER_KEYS = {
    "simulation.dt_s",
    "simulation.warmup_time_s",
    "simulation.episode_time_s",
    "simulation.num_research_vehicles",
    "simulation.num_task_vehicles",
    "simulation.num_service_vehicles",
    "service.count",
    "service.catalog_path",
    "task.arrival_probability",
    "task.popularity_zeta",
    "task.popularity_epoch_length",
    "task.popularity_changes_per_epoch",
    "task.input_relative_jitter",
    "task.load_relative_jitter",
    "nodes.sv.cache_capacity_mb",
    "nodes.rsu.cache_capacity_mb",
    "nodes.uav.cache_capacity_mb",
    "nodes.sv.eta_ca",
    "nodes.rsu.eta_ca",
    "nodes.uav.eta_ca",
    "nodes.sv.xi_ca",
    "nodes.rsu.xi_ca",
    "nodes.uav.xi_ca",
    "nodes.vehicle.local_cpu_cycles_per_s",
    "nodes.vehicle.local_cache_capacity_mb",
    "nodes.vehicle.kappa",
    "nodes.vehicle.transmit_power_w",
    "nodes.sv.cpu_cycles_per_s",
    "nodes.rsu.cpu_cycles_per_s",
    "nodes.uav.cpu_cycles_per_s",
    "nodes.cloud.cpu_cycles_per_s",
    "nodes.cloud.uplink_rate_bps",
    "nodes.cloud.cloud_delay_factor",
    "nodes.sv.bandwidth_hz",
    "nodes.rsu.bandwidth_hz",
    "nodes.uav.bandwidth_hz",
    "nodes.uav.altitude_m",
    "nodes.uav.max_horizontal_speed_mps",
    "nodes.uav.min_safety_distance_m",
    "nodes.uav.movement.movement_energy_coefficient_j_per_m",
    "nodes.uav.movement.movement_fixed_cost",
    "communication.ground.channel_gain_at_1m",
    "communication.ground.pathloss_exponent",
    "communication.ground.bandwidth_hz",
    "communication.noise.noise_power_w",
    "communication.air.eta1",
    "communication.air.eta2",
    "communication.air.carrier_frequency_hz",
    "communication.air.beta_los",
    "communication.air.beta_nlos",
    "communication.air.alpha_los",
    "communication.air.alpha_nlos",
    "communication.cloud.uplink_rate_bps",
    "objective.omega_delay",
    "objective.omega_energy",
    "objective.lambda_cache",
    "objective.lambda_movement",
    "objective.T_ref",
    "objective.E_ref",
}


def validate_for_formal_experiment(
    config: dict[str, Any],
    allow_provisional: bool = False,
) -> None:
    validate_config(config)
    unresolved = {
        key: status
        for key, status in list_unresolved_parameters(config).items()
        if key in FORMAL_SYSTEM_PARAMETER_KEYS
    }
    todo = {
        key: status
        for key, status in unresolved.items()
        if status == "TODO_LITERATURE_CALIBRATION"
    }
    provisional = {
        key: status
        for key, status in unresolved.items()
        if status == "PROVISIONAL_IMPLEMENTATION"
    }
    if todo:
        names = ", ".join(sorted(todo)[:8])
        raise ValueError(f"正式实验禁止使用尚待文献校准参数：{names}")
    if provisional and not allow_provisional:
        names = ", ".join(sorted(provisional)[:8])
        raise ValueError(
            "正式实验仍存在PROVISIONAL_IMPLEMENTATION参数；"
            f"请先校准或显式 allow_provisional=True：{names}"
        )


def _walk_numbers(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 不能是 NaN/Inf。")


def _require_positive(config: dict[str, Any], dotted_key: str) -> None:
    value = get_config_value(config, dotted_key)
    if value is None:
        raise ValueError(f"缺少配置项：{dotted_key}")
    if float(value) <= 0.0:
        raise ValueError(f"{dotted_key} 必须大于 0。")


def _require_path_exists(config: dict[str, Any], dotted_key: str) -> None:
    project_root = Path(config["_meta"]["project_root"])
    value = get_config_value(config, dotted_key)
    if not value:
        raise ValueError(f"缺少路径配置：{dotted_key}")
    path = resolve_path(project_root, value)
    if not path.exists():
        raise FileNotFoundError(f"{dotted_key} 指向的路径不存在：{path}")


def validate_config(config: dict[str, Any]) -> None:
    project_root = Path(config.get("_meta", {}).get("project_root", find_project_root()))
    config.setdefault("_meta", {})
    config["_meta"].setdefault("project_root", str(project_root))

    _walk_numbers(config)

    uav_count = int(get_config_value(config, "nodes.uav.count"))
    if uav_count < 2:
        raise ValueError("当前系统要求UAV数量至少为2，请检查system_config.toml中的nodes.uav.count。")
    uav_instances = get_config_value(config, "nodes.uav.instances", [])
    if len(uav_instances) != uav_count:
        raise ValueError("nodes.uav.instances 数量必须等于 nodes.uav.count。")
    expected_uav_ids = [f"UAV_{index}" for index in range(uav_count)]
    actual_uav_ids = [str(item.get("id")) for item in uav_instances]
    if actual_uav_ids != expected_uav_ids:
        raise ValueError("UAV ID 必须按 UAV_0, UAV_1, ... 连续命名。")

    min_safety_distance_m = float(
        get_config_value(config, "nodes.uav.min_safety_distance_m")
    )
    if min_safety_distance_m <= 0.0:
        raise ValueError("nodes.uav.min_safety_distance_m 必须大于 0。")
    for left_index, left in enumerate(uav_instances):
        for right in uav_instances[left_index + 1 :]:
            distance = math.hypot(
                float(left["initial_x_m"]) - float(right["initial_x_m"]),
                float(left["initial_y_m"]) - float(right["initial_y_m"]),
            )
            if distance + 1.0e-12 < min_safety_distance_m:
                raise ValueError(
                    "UAV 初始位置违反 nodes.uav.min_safety_distance_m："
                    f"distance={distance:.12g}, required={min_safety_distance_m:.12g}。"
                )

    roles_path = resolve_path(project_root, get_config_value(config, "paths.vehicle_roles"))
    with roles_path.open("r", encoding="utf-8") as stream:
        roles = json.load(stream)
    tv = [str(value) for value in roles.get("task_vehicles", [])]
    sv = [str(value) for value in roles.get("service_vehicles", [])]
    research = [str(value) for value in roles.get("research_vehicles", [])]
    if len(tv) + len(sv) != len(research) or set(tv) | set(sv) != set(research):
        raise ValueError("TV+SV数量或集合与research vehicle不一致。")

    if int(get_config_value(config, "prediction.vehicle.horizon")) < 1:
        raise ValueError("prediction.vehicle.horizon 必须 >= 1。")
    if int(get_config_value(config, "scenario.num_scenarios")) < 1:
        raise ValueError("scenario.num_scenarios 必须 >= 1。")
    if get_config_value(config, "algorithm.uav.method", "coarse_to_fine_coordinate") != (
        "coarse_to_fine_coordinate"
    ):
        raise ValueError("algorithm.uav.method 当前只支持 coarse_to_fine_coordinate。")
    if int(get_config_value(config, "algorithm.uav.coarse_num_angles")) < 4:
        raise ValueError("algorithm.uav.coarse_num_angles 必须 >= 4。")
    if int(get_config_value(config, "algorithm.uav.refine_num_angles")) < 4:
        raise ValueError("algorithm.uav.refine_num_angles 必须 >= 4。")
    if int(get_config_value(config, "algorithm.uav.refine_rounds")) < 0:
        raise ValueError("algorithm.uav.refine_rounds 不能为负。")
    if int(get_config_value(config, "algorithm.uav.max_coordinate_rounds")) < 1:
        raise ValueError("algorithm.uav.max_coordinate_rounds 必须 >= 1。")
    if float(get_config_value(config, "algorithm.uav.refine_radius_shrink")) <= 0.0:
        raise ValueError("algorithm.uav.refine_radius_shrink 必须大于 0。")
    if float(get_config_value(config, "algorithm.uav.position_tolerance_m")) < 0.0:
        raise ValueError("algorithm.uav.position_tolerance_m 不能为负。")
    if float(get_config_value(config, "algorithm.uav.objective_tolerance")) < 0.0:
        raise ValueError("algorithm.uav.objective_tolerance 不能为负。")
    radius_fractions = get_config_value(config, "algorithm.uav.coarse_radius_fractions")
    if not isinstance(radius_fractions, list) or not radius_fractions:
        raise ValueError("algorithm.uav.coarse_radius_fractions 必须是非空列表。")
    for value in radius_fractions:
        number = float(value)
        if number <= 0.0 or number > 1.0:
            raise ValueError("algorithm.uav.coarse_radius_fractions 必须位于 (0,1]。")
    if get_config_value(config, "algorithm.configuration.method", "cache_uav_coordinate") != (
        "cache_uav_coordinate"
    ):
        raise ValueError("algorithm.configuration.method 当前只支持 cache_uav_coordinate。")
    if int(get_config_value(config, "algorithm.configuration.max_outer_rounds")) < 1:
        raise ValueError("algorithm.configuration.max_outer_rounds 必须 >= 1。")
    if float(get_config_value(config, "algorithm.configuration.objective_tolerance")) < 0.0:
        raise ValueError("algorithm.configuration.objective_tolerance 不能为负。")
    min_cpu_fraction = float(get_config_value(config, "algorithm.kkt.min_cpu_fraction"))
    min_bandwidth_fraction = float(get_config_value(config, "algorithm.kkt.min_bandwidth_fraction"))
    if not 0.0 < min_cpu_fraction <= 1.0:
        raise ValueError("algorithm.kkt.min_cpu_fraction 必须位于 (0,1]。")
    if not 0.0 < min_bandwidth_fraction <= 1.0:
        raise ValueError("algorithm.kkt.min_bandwidth_fraction 必须位于 (0,1]。")
    if float(get_config_value(config, "algorithm.kkt.bisection_tolerance")) <= 0.0:
        raise ValueError("algorithm.kkt.bisection_tolerance 必须大于 0。")
    if int(get_config_value(config, "algorithm.kkt.max_bisection_iterations")) < 1:
        raise ValueError("algorithm.kkt.max_bisection_iterations 必须 >= 1。")
    if get_config_value(config, "algorithm.matching.method", "transfer_exchange") != (
        "transfer_exchange"
    ):
        raise ValueError("algorithm.matching.method 当前只支持 transfer_exchange。")
    if int(get_config_value(config, "algorithm.matching.max_updates")) < 1:
        raise ValueError("algorithm.matching.max_updates 必须 >= 1。")
    if float(get_config_value(config, "algorithm.matching.objective_tolerance")) < 0.0:
        raise ValueError("algorithm.matching.objective_tolerance 不能为负。")
    ppo_method = str(get_config_value(config, "algorithm.ppo.method", "trigger_actor_critic"))
    if ppo_method not in {
        "trigger_actor_critic",
        "proposed_per_trigger_q",
        "proposed_centralized_cf_q",
        "proposed_sparse_direct_cf_ppo",
        "proposed_sparse_direct_cf_frozen_policy_batch",
        "proposed_sparse_direct_cf_frozen_policy_batch_refdrift",
        "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift",
    }:
        raise ValueError(
            "algorithm.ppo.method 当前只支持 trigger_actor_critic、"
            "proposed_per_trigger_q、proposed_centralized_cf_q 或 "
            "proposed_sparse_direct_cf_ppo、proposed_sparse_direct_cf_frozen_policy_batch 或 "
            "proposed_sparse_direct_cf_frozen_policy_batch_refdrift 或 "
            "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift。"
        )
    credit_assignment = get_config_value(config, "algorithm.ppo.credit_assignment")
    if credit_assignment is not None and str(credit_assignment) not in {
        "legacy_scalar",
        "per_trigger_q",
        "centralized_counterfactual_q",
        "sparse_direct_cf",
        "hierarchical_gate_direct_cf",
    }:
        raise ValueError(
            "algorithm.ppo.credit_assignment 必须是 legacy_scalar、per_trigger_q "
            "centralized_counterfactual_q、sparse_direct_cf 或 "
            "hierarchical_gate_direct_cf。"
        )
    if ppo_method == "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift" and (
        credit_assignment is not None
        and str(credit_assignment) != "hierarchical_gate_direct_cf"
    ):
        raise ValueError(
            "hierarchical Gate method 必须使用 credit_assignment=hierarchical_gate_direct_cf。"
        )
    credit_phase = get_config_value(config, "algorithm.ppo.credit_phase")
    if credit_phase is not None and str(credit_phase) not in {"critic_warmup", "joint_training"}:
        raise ValueError(
            "algorithm.ppo.credit_phase 必须是 critic_warmup 或 joint_training。"
        )
    ppo_hidden_sizes = get_config_value(config, "algorithm.ppo.hidden_sizes", [])
    if not isinstance(ppo_hidden_sizes, list) or not ppo_hidden_sizes:
        raise ValueError("algorithm.ppo.hidden_sizes 必须是非空列表。")
    if any(int(value) <= 0 for value in ppo_hidden_sizes):
        raise ValueError("algorithm.ppo.hidden_sizes 必须全部为正整数。")
    if int(get_config_value(config, "algorithm.ppo.autoregressive_hidden_size")) <= 0:
        raise ValueError("algorithm.ppo.autoregressive_hidden_size 必须大于 0。")
    actor_architecture = str(
        get_config_value(
            config,
            "algorithm.ppo.actor_architecture",
            "v3_shared_logit_head",
        )
    )
    if actor_architecture not in {
        "v3_shared_logit_head",
        "v3_encoder_trigger_heads_frozen_ar_v1",
        "v3_full_shared_actor_frozen_policy_batch_v1",
    }:
        raise ValueError(
            "algorithm.ppo.actor_architecture 必须是 v3_shared_logit_head、"
            "v3_encoder_trigger_heads_frozen_ar_v1 或 "
            "v3_full_shared_actor_frozen_policy_batch_v1。"
        )
    ppo_gamma = float(get_config_value(config, "algorithm.ppo.gamma"))
    ppo_gae_lambda = float(get_config_value(config, "algorithm.ppo.gae_lambda"))
    if not 0.0 <= ppo_gamma <= 1.0:
        raise ValueError("algorithm.ppo.gamma 必须位于 [0,1]。")
    if not 0.0 <= ppo_gae_lambda <= 1.0:
        raise ValueError("algorithm.ppo.gae_lambda 必须位于 [0,1]。")
    ppo_clip_epsilon = float(get_config_value(config, "algorithm.ppo.clip_epsilon"))
    ppo_learning_rate = float(get_config_value(config, "algorithm.ppo.learning_rate"))
    ppo_value_coef = float(get_config_value(config, "algorithm.ppo.value_loss_coefficient"))
    ppo_entropy_coef = float(get_config_value(config, "algorithm.ppo.entropy_coefficient"))
    ppo_max_grad_norm = float(get_config_value(config, "algorithm.ppo.max_grad_norm"))
    ppo_update_epochs = int(get_config_value(config, "algorithm.ppo.update_epochs"))
    ppo_num_minibatches = int(get_config_value(config, "algorithm.ppo.num_minibatches"))
    if not 0.0 < ppo_clip_epsilon < 1.0:
        raise ValueError("algorithm.ppo.clip_epsilon 必须位于 (0,1)。")
    if ppo_learning_rate <= 0.0:
        raise ValueError("algorithm.ppo.learning_rate 必须大于 0。")
    if ppo_value_coef < 0.0:
        raise ValueError("algorithm.ppo.value_loss_coefficient 不能为负。")
    if ppo_entropy_coef < 0.0:
        raise ValueError("algorithm.ppo.entropy_coefficient 不能为负。")
    if ppo_max_grad_norm <= 0.0:
        raise ValueError("algorithm.ppo.max_grad_norm 必须大于 0。")
    if ppo_update_epochs < 1:
        raise ValueError("algorithm.ppo.update_epochs 必须 >= 1。")
    if ppo_num_minibatches < 1:
        raise ValueError("algorithm.ppo.num_minibatches 必须 >= 1。")
    cf_samples_per_episode = get_config_value(config, "algorithm.ppo.cf_samples_per_episode")
    if cf_samples_per_episode is not None:
        if isinstance(cf_samples_per_episode, bool) or int(cf_samples_per_episode) != cf_samples_per_episode:
            raise ValueError("algorithm.ppo.cf_samples_per_episode 必须是正整数。")
        if int(cf_samples_per_episode) <= 0:
            raise ValueError("algorithm.ppo.cf_samples_per_episode 必须是正整数。")
    cf_batch_episodes = get_config_value(config, "algorithm.ppo.cf_batch_episodes", 1)
    if isinstance(cf_batch_episodes, bool) or int(cf_batch_episodes) != cf_batch_episodes:
        raise ValueError("algorithm.ppo.cf_batch_episodes 必须是正整数。")
    if int(cf_batch_episodes) <= 0:
        raise ValueError("algorithm.ppo.cf_batch_episodes 必须是正整数。")
    reference_drift_cap = get_config_value(config, "algorithm.ppo.reference_probability_drift_cap")
    if ppo_method in {
        "proposed_sparse_direct_cf_frozen_policy_batch_refdrift",
        "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift",
    } and reference_drift_cap is None:
        raise ValueError(
            "refdrift正式实现必须显式设置algorithm.ppo.reference_probability_drift_cap=0.005。"
        )
    if reference_drift_cap is not None:
        cap = float(reference_drift_cap)
        if not 0.0 < cap <= 1.0:
            raise ValueError(
                "algorithm.ppo.reference_probability_drift_cap 必须位于 (0,1]。"
            )
        if ppo_method in {
            "proposed_sparse_direct_cf_frozen_policy_batch_refdrift",
            "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift",
        } and cap != 0.005:
            raise ValueError(
                "refdrift正式实现固定使用reference_probability_drift_cap=0.005。"
            )
    cf_sampling_strategy = str(
        get_config_value(config, "algorithm.ppo.cf_sampling_strategy", "uniform")
    )
    if cf_sampling_strategy not in {"uniform", "trigger_stratified"}:
        raise ValueError(
            "algorithm.ppo.cf_sampling_strategy 必须是 uniform 或 trigger_stratified。"
        )
    arrival = float(get_config_value(config, "task.arrival_probability"))
    if not 0.0 <= arrival <= 1.0:
        raise ValueError("task.arrival_probability 必须位于 [0,1]。")
    if float(get_config_value(config, "task.popularity_zeta")) <= 0.0:
        raise ValueError("task.popularity_zeta 必须大于 0。")

    for key in (
        "nodes.sv.cache_capacity_mb",
        "nodes.rsu.cache_capacity_mb",
        "nodes.uav.cache_capacity_mb",
        "nodes.vehicle.local_cache_capacity_mb",
        "nodes.vehicle.local_cpu_cycles_per_s",
        "nodes.sv.cpu_cycles_per_s",
        "nodes.rsu.cpu_cycles_per_s",
        "nodes.uav.cpu_cycles_per_s",
        "nodes.cloud.cpu_cycles_per_s",
        "nodes.uav.altitude_m",
        "nodes.uav.max_horizontal_speed_mps",
        "nodes.uav.min_safety_distance_m",
        "nodes.uav.movement.movement_energy_coefficient_j_per_m",
        "nodes.sv.bandwidth_hz",
        "nodes.rsu.bandwidth_hz",
        "nodes.uav.bandwidth_hz",
        "communication.ground.bandwidth_hz",
        "communication.air.eta1",
        "communication.air.eta2",
        "communication.air.carrier_frequency_hz",
        "communication.air.beta_los",
        "communication.air.beta_nlos",
        "communication.air.alpha_los",
        "communication.air.alpha_nlos",
    ):
        _require_positive(config, key)

    rsu_count = int(get_config_value(config, "nodes.rsu.count"))
    rsu_instances = get_config_value(config, "nodes.rsu.instances", [])
    if len(rsu_instances) != rsu_count:
        raise ValueError("nodes.rsu.instances 数量必须等于 nodes.rsu.count。")

    omega_delay = float(get_config_value(config, "objective.omega_delay"))
    omega_energy = float(get_config_value(config, "objective.omega_energy"))
    if omega_delay < 0.0 or omega_energy < 0.0 or omega_delay + omega_energy <= 0.0:
        raise ValueError("objective omega 权重必须非负且总和大于 0。")
    _require_positive(config, "objective.T_ref")
    _require_positive(config, "objective.E_ref")
    cloud_delay_factor = float(get_config_value(config, "nodes.cloud.cloud_delay_factor"))
    if cloud_delay_factor < 1.0:
        raise ValueError(
            "Cloud传输扩展系数psi_c必须不小于1，"
            "因为其用于表示蜂窝接入基础上传时延之外的"
            "回传网络和核心网附加时延。"
        )

    # 单位/重复参数一致性：配置中统一使用 cycle/s、Hz、W、bit/s、MB。
    if float(get_config_value(config, "communication.ground.bandwidth_hz")) != float(
        get_config_value(config, "nodes.sv.bandwidth_hz")
    ):
        raise ValueError("communication.ground.bandwidth_hz 必须与 nodes.sv.bandwidth_hz 一致。")
    if float(get_config_value(config, "communication.ground.bandwidth_hz")) != float(
        get_config_value(config, "nodes.rsu.bandwidth_hz")
    ):
        raise ValueError("communication.ground.bandwidth_hz 必须与 nodes.rsu.bandwidth_hz 一致。")
    if float(get_config_value(config, "communication.cloud.uplink_rate_bps")) != float(
        get_config_value(config, "nodes.cloud.uplink_rate_bps")
    ):
        raise ValueError("communication.cloud.uplink_rate_bps 必须与 nodes.cloud.uplink_rate_bps 一致。")

    for key, status in config.get("parameter_status", {}).items():
        if status not in {
            "CONFIRMED_DATA",
            "PROVISIONAL_IMPLEMENTATION",
            "TODO_LITERATURE_CALIBRATION",
        }:
            raise ValueError(f"未知参数状态 {status}：{key}")

    for key in (
        "paths.service_catalog",
        "paths.vehicle_roles",
        "paths.prediction_scenarios",
        "paths.sumo_net",
        "paths.residual_bank",
        "paths.popularity_gru",
        "paths.tv_local_cache",
    ):
        _require_path_exists(config, key)


def main() -> int:
    config = load_config()
    print("system_config.toml 解析与校验通过")
    print(f"config_hash={config_hash(config)}")
    print(f"uav_count={get_config_value(config, 'nodes.uav.count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
