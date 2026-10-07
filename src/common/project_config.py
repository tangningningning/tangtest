#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一项目配置读取、override、校验和稳定 hash。

本文件只处理配置，不放业务算法逻辑。Python 3.11+ 使用标准库 tomllib；
若运行环境低于 3.11，则尝试使用轻量 tomli。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only for Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]


def find_project_root() -> Path:
    candidates = (Path(__file__).resolve().parents[2], Path.cwd().resolve())
    for candidate in candidates:
        if (candidate / "configs" / "proposed" / "system_config.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("无法找到包含 configs/proposed/system_config.toml 的项目根目录。")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def parse_toml_file(path: str | Path) -> dict[str, Any]:
    """解析 TOML 文件，不注入运行时元数据或执行配置校验。"""

    config_path = Path(path).expanduser()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"TOML 配置根对象必须是 table：{config_path}")
    return config


def semantic_config_hash(config: dict[str, Any]) -> str:
    """对解析后的配置值计算稳定哈希，并忽略运行时注入的元数据。"""

    if not isinstance(config, dict):
        raise TypeError("语义配置哈希要求输入解析后的配置字典。")
    semantic_config = {
        key: value for key, value in config.items() if key != "_meta"
    }
    canonical = json.dumps(
        semantic_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def raw_config_sha256(path: str | Path) -> str:
    """计算 TOML 文件原始字节哈希，仅用于来源记录。"""

    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def load_config(
    path: str | Path = "configs/proposed/system_config.toml",
    *,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    project_root = find_project_root()
    config_path = resolve_path(project_root, path)
    config = parse_toml_file(config_path)
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    validate_config(config, require_artifacts=require_artifacts)
    return config


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED = "full_episode_randomized"


def get_ppo_data_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """读取并校验正式的完整 episode 数据协议。"""

    raw_protocol = get_config_value(config, "algorithm.ppo.data_protocol")
    if not isinstance(raw_protocol, dict):
        raise ValueError("缺少 algorithm.ppo.data_protocol 配置表。")
    protocol = str(raw_protocol.get("protocol", ""))
    if protocol != PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError(
            "algorithm.ppo.data_protocol.protocol 必须为 full_episode_randomized。"
        )

    dt_s = float(get_config_value(config, "simulation.dt_s"))
    episode_time_s = float(get_config_value(config, "simulation.episode_time_s"))
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("simulation.dt_s 必须是正的有限数。")
    if not math.isfinite(episode_time_s) or episode_time_s <= 0.0:
        raise ValueError("simulation.episode_time_s 必须是正的有限数。")
    physical_episode_slots_f = episode_time_s / dt_s
    physical_episode_slots = int(round(physical_episode_slots_f))
    if physical_episode_slots <= 0 or not math.isclose(
        physical_episode_slots_f,
        float(physical_episode_slots),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("simulation.episode_time_s / simulation.dt_s 必须是正整数。")

    def read_int(key: str, *, required: bool = True) -> int | None:
        if key not in raw_protocol:
            if required:
                raise ValueError(
                    f"algorithm.ppo.data_protocol 缺少 {key}。"
                )
            return None
        value = raw_protocol[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"algorithm.ppo.data_protocol.{key} 必须是整数。"
            )
        return int(value)

    train_start = read_int("train_origin_start")
    assert train_start is not None
    if train_start != 0:
        raise ValueError(
            "algorithm.ppo.data_protocol.train_origin_start 必须为正式 origin 的起点0。"
        )

    num_minibatches_value = get_config_value(
        config, "algorithm.ppo.num_minibatches"
    )
    if (
        isinstance(num_minibatches_value, bool)
        or not isinstance(num_minibatches_value, int)
        or int(num_minibatches_value) <= 0
    ):
        raise ValueError("algorithm.ppo.num_minibatches 必须是正整数。")
    num_minibatches = int(num_minibatches_value)

    configured_episode_slots = read_int("episode_slots")
    configured_train_end = read_int("train_origin_end")
    configured_rollout_steps = read_int("rollout_steps")
    configured_terminal = read_int("terminal_origin")
    assert (
        configured_episode_slots is not None
        and configured_train_end is not None
        and configured_rollout_steps is not None
        and configured_terminal is not None
    )
    if configured_episode_slots != physical_episode_slots:
        raise ValueError(
            "full_episode_randomized episode_slots 必须等于 "
            "simulation.episode_time_s / simulation.dt_s："
            f"configured={configured_episode_slots}, "
            f"expected={physical_episode_slots}。"
        )
    if configured_train_end != configured_episode_slots - 1:
        raise ValueError(
            "full_episode_randomized train_origin_end 必须等于 episode_slots-1。"
        )
    if configured_rollout_steps != configured_episode_slots:
        raise ValueError(
            "full_episode_randomized rollout_steps 必须等于 episode_slots。"
        )
    if configured_terminal != configured_episode_slots - 1:
        raise ValueError(
            "full_episode_randomized terminal_origin 必须等于 episode_slots-1。"
        )
    if "test_origin_start" in raw_protocol or "test_origin_end" in raw_protocol:
        raise ValueError(
            "full_episode_randomized 不允许设置同一 realization 内的 test split。"
        )
    train_end = configured_train_end
    episode_slots = configured_episode_slots
    configured_rollout = configured_rollout_steps
    terminal_origin = configured_terminal
    test_start = None
    test_end = None

    train_origin_count = train_end - train_start + 1
    if train_origin_count <= 0 or train_origin_count % num_minibatches != 0:
        raise ValueError(
            "PPO train origin 数量必须为正并可被 algorithm.ppo.num_minibatches 整除："
            f"count={train_origin_count}, num_minibatches={num_minibatches}。"
        )
    configured_minibatch_size = get_config_value(
        config, "algorithm.ppo.minibatch_size", None
    )
    if configured_minibatch_size is None:
        minibatch_size = train_origin_count // num_minibatches
    else:
        if (
            isinstance(configured_minibatch_size, bool)
            or not isinstance(configured_minibatch_size, int)
            or int(configured_minibatch_size) <= 0
        ):
            raise ValueError("algorithm.ppo.minibatch_size 必须是正整数。")
        minibatch_size = int(configured_minibatch_size)
        expected_minibatch_size = train_origin_count // num_minibatches
        if minibatch_size != expected_minibatch_size:
            raise ValueError(
                "algorithm.ppo.minibatch_size 必须等于完整 train episode 的 "
                "transitions / num_minibatches："
                f"configured={minibatch_size}, expected={expected_minibatch_size}。"
            )

    return {
        "protocol": protocol,
        "physical_episode_slots": physical_episode_slots,
        "episode_slots": episode_slots,
        "train_origin_start": train_start,
        "train_origin_end": train_end,
        "test_origin_start": test_start,
        "test_origin_end": test_end,
        "rollout_steps": configured_rollout,
        "terminal_origin": terminal_origin,
        "num_minibatches": num_minibatches,
        "minibatch_size": minibatch_size,
    }


def _node_positions_path(config: dict[str, Any]) -> Path:
    project_root = Path(
        config.get("_meta", {}).get("project_root", find_project_root())
    )
    value = get_config_value(config, "paths.node_positions")
    if not value:
        raise ValueError("缺少路径配置：paths.node_positions")
    return resolve_path(project_root, value)


def _normalise_node_position(node_id: str, value: Any) -> list[float]:
    """读取正式 node_positions.json 中对象格式的 position 字段。"""

    if not isinstance(value, dict) or "position" not in value:
        raise ValueError(
            f"node_positions.json 的 {node_id} 必须是包含 position 字段的对象。"
        )
    value = value["position"]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"node_positions.json 的 {node_id} 位置必须是 [x, y, z]。"
        )
    try:
        position = [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"node_positions.json 的 {node_id} 位置必须全部是数值。"
        ) from exc
    if not all(math.isfinite(component) for component in position):
        raise ValueError(f"node_positions.json 的 {node_id} 位置不能是 NaN/Inf。")
    return position


def load_node_positions(config: dict[str, Any]) -> dict[str, Any]:
    """读取并校验正式 node_positions.json 节点位置文件。"""

    positions_path = _node_positions_path(config)
    with positions_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("node_positions.json 根对象必须是 object。")

    raw_boundary = payload.get("network_boundary")
    if not isinstance(raw_boundary, dict):
        raise ValueError("node_positions.json 缺少 network_boundary。")
    required_boundary_keys = ("min_x", "max_x", "min_y", "max_y")
    if any(key not in raw_boundary for key in required_boundary_keys):
        raise ValueError(
            "node_positions.json 的 network_boundary 必须包含 min_x/max_x/min_y/max_y。"
        )
    boundary = {
        key: float(raw_boundary[key])
        for key in required_boundary_keys
    }
    if not all(math.isfinite(value) for value in boundary.values()):
        raise ValueError("node_positions.json 的 network_boundary 不能是 NaN/Inf。")
    if boundary["min_x"] >= boundary["max_x"] or boundary["min_y"] >= boundary["max_y"]:
        raise ValueError("node_positions.json 的 network_boundary 非法。")

    positions: dict[str, Any] = {"network_boundary": boundary}
    for role in ("rsu", "uav"):
        raw_nodes = payload.get(role)
        if not isinstance(raw_nodes, dict):
            raise ValueError(f"node_positions.json 缺少 {role} 节点位置表。")
        normalised: dict[str, list[float]] = {}
        for raw_id, raw_position in raw_nodes.items():
            node_id = str(raw_id)
            position = _normalise_node_position(node_id, raw_position)
            x, y = position[:2]
            if not (
                boundary["min_x"] <= x <= boundary["max_x"]
                and boundary["min_y"] <= y <= boundary["max_y"]
            ):
                raise ValueError(f"{node_id} 坐标超出 SUMO 路网区域：{position}")
            normalised[node_id] = position
        positions[role] = normalised
    return positions


def _materialise_node_instances(config: dict[str, Any]) -> None:
    """将 JSON 位置转换为现有模块使用的 nodes.*.instances 结构。"""

    positions = load_node_positions(config)
    nodes = config.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("缺少 nodes 配置表。")

    for role, prefix, x_key, y_key, z_key, altitude_key in (
        ("rsu", "RSU", "x_m", "y_m", "z_m", "deployment_altitude_m"),
        (
            "uav",
            "UAV",
            "initial_x_m",
            "initial_y_m",
            "initial_z_m",
            "deployment_altitude_m",
        ),
    ):
        count_value = get_config_value(config, f"nodes.{role}.count")
        if (
            count_value is None
            or isinstance(count_value, bool)
            or int(count_value) != count_value
            or int(count_value) < 1
        ):
            raise ValueError(f"nodes.{role}.count 必须是正整数。")
        count = int(count_value)
        expected_ids = [f"{prefix}_{index}" for index in range(count)]
        role_positions = positions[role]
        if set(role_positions) != set(expected_ids):
            raise ValueError(
                f"node_positions.json 的 {role} IDs 必须与 nodes.{role}.count 一致。"
            )
        instances = []
        expected_altitude = get_config_value(
            config,
            f"nodes.{role}.{altitude_key}",
        )
        if expected_altitude is None:
            raise ValueError(f"缺少配置项：nodes.{role}.{altitude_key}")
        expected_altitude = float(expected_altitude)
        for node_id in expected_ids:
            x, y, z = role_positions[node_id]
            if not math.isclose(z, expected_altitude, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{node_id} 高度 {z} 与配置 nodes.{role}.{altitude_key} "
                    f"={expected_altitude} 不一致。"
                )
            instances.append(
                {
                    "id": node_id,
                    x_key: x,
                    y_key: y,
                    z_key: z,
                }
            )
        node_config = nodes.get(role)
        if not isinstance(node_config, dict):
            raise ValueError(f"缺少 nodes.{role} 配置表。")
        node_config["instances"] = instances










def list_unresolved_parameters(config: dict[str, Any]) -> dict[str, str]:
    statuses = config.get("parameter_status", {})
    allowed = {
        "CONFIRMED_DATA",
        "TRAIN_ONLY_CALIBRATION",
        "EXPERIMENT_DESIGN",
        "ALGORITHM_DESIGN",
        "LITERATURE_DERIVED",
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
    "execution.cross_slot_task_model",
    "simulation.warmup_time_s",
    "simulation.episode_time_s",
    "simulation.num_research_vehicles",
    "simulation.num_task_vehicles",
    "simulation.num_service_vehicles",
    "service.count",
    "service.catalog_path",
    "service.generation.service_size_min_gb",
    "service.generation.service_size_max_gb",
    "service.generation.input_size_min_kb",
    "service.generation.input_size_max_kb",
    "service.generation.cycles_per_bit_min",
    "service.generation.cycles_per_bit_max",
    "task.arrival_probability",
    "task.popularity_zeta",
    "task.popularity_epoch_length",
    "task.popularity_changes_per_epoch",
    "task.input_relative_jitter",
    "task.load_relative_jitter",
    "task.master_seed",
    "nodes.sv.cache_capacity_mb",
    "nodes.rsu.count",
    "nodes.rsu.cache_capacity_mb",
    "nodes.uav.count",
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
    "nodes.cloud.total_cpu_cycles_per_s",
    "nodes.cloud.uplink_rate_bps",
    "nodes.cloud.total_bandwidth_bps",
    "nodes.cloud.resource_sharing",
    "nodes.cloud.cloud_delay_factor",
    "nodes.sv.bandwidth_hz",
    "nodes.rsu.bandwidth_hz",
    "nodes.uav.bandwidth_hz",
    "nodes.uav.altitude_m",
    "nodes.uav.deployment_altitude_m",
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


def validate_config(
    config: dict[str, Any],
    *,
    require_artifacts: bool = True,
) -> None:
    project_root = Path(config.get("_meta", {}).get("project_root", find_project_root()))
    config.setdefault("_meta", {})
    config["_meta"].setdefault("project_root", str(project_root))

    if require_artifacts:
        _materialise_node_instances(config)
    _walk_numbers(config)

    service_count_value = get_config_value(config, "service.count")
    if (
        service_count_value is None
        or isinstance(service_count_value, bool)
        or int(service_count_value) != service_count_value
        or int(service_count_value) <= 0
    ):
        raise ValueError("service.count 必须是正整数。")
    service_generation = get_config_value(config, "service.generation")
    if not isinstance(service_generation, dict):
        raise ValueError("缺少 service.generation 配置表。")
    service_ranges = (
        ("service_size_min_gb", "service_size_max_gb", "service_size"),
        (
            "input_size_min_kb",
            "input_size_max_kb",
            "input_size",
        ),
        ("cycles_per_bit_min", "cycles_per_bit_max", "cycles_per_bit"),
    )
    for lower_key, upper_key, label in service_ranges:
        lower = service_generation.get(lower_key)
        upper = service_generation.get(upper_key)
        if lower is None or upper is None:
            raise ValueError(
                f"service.generation 缺少 {lower_key}/{upper_key}。"
            )
        lower = float(lower)
        upper = float(upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"service.generation.{label} 范围不能是 NaN/Inf。")
        if lower <= 0.0:
            raise ValueError(f"service.generation.{lower_key} 必须大于 0。")
        if upper < lower:
            raise ValueError(
                f"service.generation.{upper_key} 必须大于等于 {lower_key}。"
            )

    uav_count = int(get_config_value(config, "nodes.uav.count"))
    if uav_count < 1:
        raise ValueError("nodes.uav.count 必须至少为1，请检查 configs/proposed/system_config.toml。")
    if require_artifacts:
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
        configured_research = int(
            get_config_value(config, "simulation.num_research_vehicles")
        )
        configured_sv = int(get_config_value(config, "simulation.num_service_vehicles"))
        configured_tv = int(get_config_value(config, "simulation.num_task_vehicles"))
        if (len(research), len(sv), len(tv)) != (
            configured_research,
            configured_sv,
            configured_tv,
        ):
            raise ValueError(
                "vehicle_roles.json 车辆数量必须与 configs/proposed/system_config.toml 的正式配置一致。"
            )
        if len(research) != len(set(research)) or len(sv) != len(set(sv)) or len(tv) != len(set(tv)):
            raise ValueError("vehicle_roles.json 研究车辆角色列表不能包含重复 ID。")
        if len(tv) + len(sv) != len(research) or set(tv) | set(sv) != set(research):
            raise ValueError("TV+SV数量或集合与research vehicle不一致。")

    if int(get_config_value(config, "prediction.vehicle.horizon")) < 1:
        raise ValueError("prediction.vehicle.horizon 必须 >= 1。")
    if int(get_config_value(config, "scenario.num_scenarios")) < 1:
        raise ValueError("scenario.num_scenarios 必须 >= 1。")
    uav_method = str(
        get_config_value(config, "algorithm.uav.method", "coarse_to_fine_coordinate")
    )
    if uav_method not in {
        "coarse_to_fine_coordinate",
        "demand_aware_target_position",
    }:
        raise ValueError(
            "algorithm.uav.method 当前只支持 coarse_to_fine_coordinate/"
            "demand_aware_target_position。"
        )
    if int(get_config_value(config, "algorithm.uav.coarse_num_angles")) < 4:
        raise ValueError("algorithm.uav.coarse_num_angles 必须 >= 4。")
    if int(get_config_value(config, "algorithm.uav.refine_num_angles")) < 4:
        raise ValueError("algorithm.uav.refine_num_angles 必须 >= 4。")
    exploration_num_angles = get_config_value(
        config, "algorithm.uav.exploration_num_angles"
    )
    if exploration_num_angles is None:
        raise ValueError("缺少必需配置：algorithm.uav.exploration_num_angles。")
    if int(exploration_num_angles) < 4:
        raise ValueError("algorithm.uav.exploration_num_angles 必须 >= 4。")
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
    exploration_fractions = get_config_value(
        config, "algorithm.uav.exploration_radius_fractions"
    )
    if exploration_fractions is None:
        raise ValueError("缺少必需配置：algorithm.uav.exploration_radius_fractions。")
    if not isinstance(exploration_fractions, list) or not exploration_fractions:
        raise ValueError(
            "algorithm.uav.exploration_radius_fractions 必须是非空列表。"
        )
    if any(float(value) <= 0.0 or float(value) > 1.0 for value in exploration_fractions):
        raise ValueError(
            "algorithm.uav.exploration_radius_fractions 必须位于 (0,1]。"
        )
    demand_anchor_scales = get_config_value(
        config, "algorithm.uav.demand_anchor_scales"
    )
    if demand_anchor_scales is None:
        raise ValueError("缺少必需配置：algorithm.uav.demand_anchor_scales。")
    if not isinstance(demand_anchor_scales, list) or not demand_anchor_scales:
        raise ValueError("algorithm.uav.demand_anchor_scales 必须是非空列表。")
    if any(float(value) <= 0.0 or float(value) > 1.0 for value in demand_anchor_scales):
        raise ValueError("algorithm.uav.demand_anchor_scales 必须位于 (0,1]。")
    target_candidate_capacity = get_config_value(
        config, "algorithm.uav.target_candidate_capacity"
    )
    if target_candidate_capacity is None:
        raise ValueError("缺少必需配置：algorithm.uav.target_candidate_capacity。")
    if int(target_candidate_capacity) < 1:
        raise ValueError("algorithm.uav.target_candidate_capacity 必须 >= 1。")
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
    ppo_method = str(get_config_value(config, "algorithm.ppo.method", ""))
    if ppo_method != "PROPOSED":
        raise ValueError("algorithm.ppo.method 必须为 PROPOSED。")
    configuration_control_mode = str(
        get_config_value(config, "algorithm.ppo.configuration_control_mode", "")
    )
    if configuration_control_mode != "macro_cache_smdp_micro_uav_candidate":
        raise ValueError(
            "configuration_control_mode 必须为 macro_cache_smdp_micro_uav_candidate。"
        )
    policy_schema = str(get_config_value(config, "algorithm.ppo.policy_schema", ""))
    if policy_schema != (
        "adaptive_dual_timescale_proposed_cache_adaptive_holding_cardinality_corrected_ppo"
    ):
        raise ValueError("algorithm.ppo.policy_schema 必须为当前 PROPOSED 正式策略 schema。")
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
        actual = get_config_value(config, f"algorithm.ppo.{key}", None)
        if actual is not expected:
            raise ValueError(f"algorithm.ppo.{key} 必须为 {expected!r}。")
    prediction_horizon = int(get_config_value(config, "prediction.vehicle.horizon"))
    adaptive_holding_enabled = get_config_value(
        config, "algorithm.ppo.adaptive_holding_enabled", None
    )
    if adaptive_holding_enabled is not True:
        raise ValueError("algorithm.ppo.adaptive_holding_enabled 必须为 true。")
    adaptive_holding_min_slots = get_config_value(
        config, "algorithm.ppo.adaptive_holding_min_slots", None
    )
    adaptive_holding_max_slots = get_config_value(
        config, "algorithm.ppo.adaptive_holding_max_slots", None
    )
    for key, value in {
        "adaptive_holding_min_slots": adaptive_holding_min_slots,
        "adaptive_holding_max_slots": adaptive_holding_max_slots,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"algorithm.ppo.{key} 必须为正整数。")
    if adaptive_holding_min_slots > adaptive_holding_max_slots:
        raise ValueError(
            "algorithm.ppo.adaptive_holding_min_slots 不得大于 adaptive_holding_max_slots。"
        )
    if adaptive_holding_max_slots > prediction_horizon:
        raise ValueError("adaptive_holding_max_slots 不得超过当前预测 horizon。")
    if adaptive_holding_min_slots != 1 or adaptive_holding_max_slots != 5:
        raise ValueError(
            "PROPOSED adaptive holding 正式配置必须固定 min_slots=1、max_slots=5。"
        )

    if "target_kl" in get_config_value(config, "algorithm.ppo", {}):
        raise ValueError(
            "algorithm.ppo.target_kl 已废弃；请分别配置 macro_target_kl 和 micro_target_kl。"
        )
    macro_target_kl = get_config_value(config, "algorithm.ppo.macro_target_kl", None)
    micro_target_kl = get_config_value(config, "algorithm.ppo.micro_target_kl", None)
    for key, value in {
        "macro_target_kl": macro_target_kl,
        "micro_target_kl": micro_target_kl,
    }.items():
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"algorithm.ppo.{key} 必须为正的有限数。")
    if not math.isclose(float(macro_target_kl), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PROPOSED algorithm.ppo.macro_target_kl 必须固定为 0.10。")
    if not math.isclose(float(micro_target_kl), 0.02, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PROPOSED algorithm.ppo.micro_target_kl 必须固定为 0.02。")
    group_probability_soft_cap = get_config_value(
        config, "algorithm.ppo.micro_group_probability_soft_cap", None
    )
    if (
        group_probability_soft_cap is None
        or not math.isfinite(float(group_probability_soft_cap))
        or not 0.5 < float(group_probability_soft_cap) < 1.0
    ):
        raise ValueError(
            "algorithm.ppo.micro_group_probability_soft_cap 必须为 (0.5,1) 内的有限数。"
        )
    if not math.isclose(
        float(group_probability_soft_cap), 0.90, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "PROPOSED algorithm.ppo.micro_group_probability_soft_cap 必须固定为 0.90。"
        )
    group_entropy_floor_coefficient = get_config_value(
        config, "algorithm.ppo.micro_group_entropy_floor_coefficient", None
    )
    if (
        group_entropy_floor_coefficient is None
        or not math.isfinite(float(group_entropy_floor_coefficient))
        or float(group_entropy_floor_coefficient) < 0.0
    ):
        raise ValueError(
            "algorithm.ppo.micro_group_entropy_floor_coefficient 必须为非负有限数。"
        )
    if not math.isclose(
        float(group_entropy_floor_coefficient), 0.05, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "PROPOSED algorithm.ppo.micro_group_entropy_floor_coefficient 必须固定为 0.05。"
        )
    ppo_hidden_sizes = get_config_value(config, "algorithm.ppo.hidden_sizes", [])
    if not isinstance(ppo_hidden_sizes, list) or not ppo_hidden_sizes:
        raise ValueError("algorithm.ppo.hidden_sizes 必须是非空列表。")
    if any(isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in ppo_hidden_sizes):
        raise ValueError("algorithm.ppo.hidden_sizes 必须全部为正整数。")
    ppo_gamma = float(get_config_value(config, "algorithm.ppo.gamma"))
    ppo_gae_lambda = float(get_config_value(config, "algorithm.ppo.gae_lambda"))
    if not 0.0 <= ppo_gamma <= 1.0 or not 0.0 <= ppo_gae_lambda <= 1.0:
        raise ValueError("gamma 和 gae_lambda 必须位于 [0,1]。")
    ppo_clip_epsilon = float(get_config_value(config, "algorithm.ppo.clip_epsilon"))
    ppo_learning_rate = float(get_config_value(config, "algorithm.ppo.learning_rate"))
    ppo_actor_learning_rate = float(
        get_config_value(config, "algorithm.ppo.actor_learning_rate", ppo_learning_rate)
    )
    ppo_critic_learning_rate = float(
        get_config_value(config, "algorithm.ppo.critic_learning_rate", ppo_learning_rate)
    )
    ppo_value_coef = float(get_config_value(config, "algorithm.ppo.value_loss_coefficient"))
    ppo_entropy_coef = float(get_config_value(config, "algorithm.ppo.entropy_coefficient"))
    ppo_max_grad_norm = float(get_config_value(config, "algorithm.ppo.max_grad_norm"))
    ppo_update_epochs = int(get_config_value(config, "algorithm.ppo.update_epochs"))
    ppo_num_minibatches = int(get_config_value(config, "algorithm.ppo.num_minibatches"))
    if not 0.0 < ppo_clip_epsilon < 1.0:
        raise ValueError("algorithm.ppo.clip_epsilon 必须位于 (0,1)。")
    for key, value in {
        "learning_rate": ppo_learning_rate,
        "actor_learning_rate": ppo_actor_learning_rate,
        "critic_learning_rate": ppo_critic_learning_rate,
        "max_grad_norm": ppo_max_grad_norm,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"algorithm.ppo.{key} 必须为正的有限数。")
    realizations_per_policy_update = get_config_value(
        config, "algorithm.ppo.realizations_per_policy_update", None
    )
    if (
        isinstance(realizations_per_policy_update, bool)
        or not isinstance(realizations_per_policy_update, int)
        or int(realizations_per_policy_update) != 4
    ):
        raise ValueError("algorithm.ppo.realizations_per_policy_update 必须固定为 4。")
    macro_actor_learning_rate = float(
        get_config_value(config, "algorithm.ppo.macro_actor_learning_rate", math.nan)
    )
    macro_actor_final_learning_rate = float(
        get_config_value(config, "algorithm.ppo.macro_actor_final_learning_rate", math.nan)
    )
    macro_actor_learning_rate_schedule = str(
        get_config_value(config, "algorithm.ppo.macro_actor_learning_rate_schedule", "")
    )
    if (
        not math.isfinite(macro_actor_learning_rate)
        or not math.isfinite(macro_actor_final_learning_rate)
        or macro_actor_learning_rate <= 0.0
        or macro_actor_final_learning_rate <= 0.0
        or macro_actor_final_learning_rate > macro_actor_learning_rate
        or macro_actor_learning_rate_schedule != "cosine"
    ):
        raise ValueError(
            "PROPOSED Macro Actor LR endpoints must be positive and finite, "
            "with final <= initial and the cosine schedule."
        )
    if ppo_value_coef < 0.0 or ppo_entropy_coef < 0.0:
        raise ValueError("value_loss_coefficient 和 entropy_coefficient 不能为负。")
    if not math.isclose(ppo_entropy_coef, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("PROPOSED algorithm.ppo.entropy_coefficient 必须固定为 0。")
    if ppo_update_epochs < 1 or ppo_num_minibatches < 1:
        raise ValueError("update_epochs 和 num_minibatches 必须为正整数。")
    if int(get_config_value(config, "algorithm.ppo.minibatch_size")) <= 0:
        raise ValueError("algorithm.ppo.minibatch_size 必须为正整数。")
    prediction_batch_parallel_workers = get_config_value(
        config,
        "algorithm.configuration.prediction_batch_parallel_workers",
        1,
    )
    if (
        isinstance(prediction_batch_parallel_workers, bool)
        or int(prediction_batch_parallel_workers) != prediction_batch_parallel_workers
        or int(prediction_batch_parallel_workers) <= 0
    ):
        raise ValueError(
            "algorithm.configuration.prediction_batch_parallel_workers 必须是正整数。"
        )
    arrival = float(get_config_value(config, "task.arrival_probability"))
    if not 0.0 <= arrival <= 1.0:
        raise ValueError("task.arrival_probability 必须位于 [0,1]。")
    if float(get_config_value(config, "task.popularity_zeta")) <= 0.0:
        raise ValueError("task.popularity_zeta 必须大于 0。")
    dt_s = float(get_config_value(config, "simulation.dt_s"))
    warmup_s = float(get_config_value(config, "simulation.warmup_time_s"))
    episode_s = float(get_config_value(config, "simulation.episode_time_s"))
    if dt_s <= 0.0 or warmup_s < 0.0 or episode_s <= 0.0:
        raise ValueError("simulation.dt_s/warmup_time_s/episode_time_s 数值范围无效。")
    if str(get_config_value(config, "execution.cross_slot_task_model")) != "independent_per_slot":
        raise ValueError(
            "execution.cross_slot_task_model 必须为 independent_per_slot。"
        )
    for label, duration in (("warmup_time_s", warmup_s), ("episode_time_s", episode_s)):
        steps = duration / dt_s
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"simulation.{label} 必须是 dt_s 的整数倍。")

    # 当前 PPO 使用完整 episode 随机化协议，
    # 并在协议校验中验证 episode 终止位置。
    get_ppo_data_protocol(config)
    research_count = int(get_config_value(config, "simulation.num_research_vehicles"))
    service_count = int(get_config_value(config, "simulation.num_service_vehicles"))
    task_count = int(get_config_value(config, "simulation.num_task_vehicles"))
    if research_count <= 0 or service_count < 0 or task_count < 0:
        raise ValueError("simulation 车辆数量范围无效。")
    if service_count + task_count != research_count:
        raise ValueError("simulation SV+TV 数量必须等于 research vehicle 数量。")
    departure_begin = float(
        get_config_value(config, "simulation.random_trips_departure_begin_s", 0.0)
    )
    departure_end = float(
        get_config_value(config, "simulation.random_trips_departure_end_s", 60.0)
    )
    if departure_begin < 0.0 or departure_end <= departure_begin:
        raise ValueError("randomTrips departure begin/end 配置无效。")
    if float(get_config_value(config, "simulation.random_trips_period_s", 1.0)) <= 0.0:
        raise ValueError("simulation.random_trips_period_s 必须大于 0。")
    if float(get_config_value(config, "simulation.random_trips_min_distance_m", 0.0)) < 0.0:
        raise ValueError("simulation.random_trips_min_distance_m 不能为负数。")
    if int(get_config_value(config, "simulation.random_trips_intermediate", 0)) < 0:
        raise ValueError("simulation.random_trips_intermediate 不能为负数。")
    if int(get_config_value(config, "simulation.random_trips_maxtries", 1)) <= 0:
        raise ValueError("simulation.random_trips_maxtries 必须大于 0。")

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
        "nodes.cloud.total_cpu_cycles_per_s",
        "nodes.cloud.total_bandwidth_bps",
        "nodes.uav.altitude_m",
        "nodes.uav.max_horizontal_speed_mps",
        "nodes.uav.min_safety_distance_m",
        "nodes.uav.movement.movement_energy_coefficient_j_per_m",
        "nodes.sv.bandwidth_hz",
        "nodes.rsu.bandwidth_hz",
        "nodes.uav.bandwidth_hz",
        "nodes.rsu.deployment_altitude_m",
        "nodes.uav.deployment_altitude_m",
        "nodes.uav.deployment_offset_fraction",
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
    if rsu_count < 1:
        raise ValueError("nodes.rsu.count 必须至少为1。")
    if require_artifacts:
        rsu_instances = get_config_value(config, "nodes.rsu.instances", [])
        if len(rsu_instances) != rsu_count:
            raise ValueError("nodes.rsu.instances 数量必须等于 nodes.rsu.count。")
        expected_rsu_ids = [f"RSU_{index}" for index in range(rsu_count)]
        actual_rsu_ids = [str(item.get("id")) for item in rsu_instances]
        if actual_rsu_ids != expected_rsu_ids:
            raise ValueError("RSU ID 必须按 RSU_0, RSU_1, ... 连续命名。")

    omega_delay = float(get_config_value(config, "objective.omega_delay"))
    omega_energy = float(get_config_value(config, "objective.omega_energy"))
    if not 0.0 <= omega_delay <= 1.0:
        raise ValueError("objective.omega_delay 必须位于 [0,1]。")
    if not 0.0 <= omega_energy <= 1.0:
        raise ValueError("objective.omega_energy 必须位于 [0,1]。")
    if not math.isclose(omega_delay + omega_energy, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "objective.omega_delay 和 objective.omega_energy "
            "必须位于 [0,1] 且总和等于 1。"
        )
    raw_uav_movement_cost_weight = get_config_value(
        config, "objective.uav_movement_cost_weight"
    )
    if raw_uav_movement_cost_weight is None:
        raise ValueError("缺少必需配置：objective.uav_movement_cost_weight。")
    uav_movement_cost_weight = float(raw_uav_movement_cost_weight)
    if not math.isfinite(uav_movement_cost_weight) or uav_movement_cost_weight < 0.0:
        raise ValueError("objective.uav_movement_cost_weight 必须是非负有限数。")
    cache_cost_weight = float(
        get_config_value(config, "objective.cache_cost_weight", 1.0)
    )
    if not math.isfinite(cache_cost_weight) or cache_cost_weight < 0.0:
        raise ValueError("objective.cache_cost_weight 必须是非负有限数。")
    _require_positive(config, "objective.T_ref")
    _require_positive(config, "objective.E_ref")
    cloud_delay_factor = float(get_config_value(config, "nodes.cloud.cloud_delay_factor"))
    if cloud_delay_factor < 1.0:
        raise ValueError(
            "Cloud传输扩展系数psi_c必须不小于1，"
            "因为其用于表示蜂窝接入基础上传时延之外的"
            "回传网络和核心网附加时延。"
        )
    cloud_resource_sharing = get_config_value(config, "nodes.cloud.resource_sharing")
    if not isinstance(cloud_resource_sharing, bool):
        raise ValueError(
            "nodes.cloud.resource_sharing 必须显式配置为 true/false，"
            "不允许缺失后静默采用默认共享语义。"
        )
    if cloud_resource_sharing:
        raise ValueError(
            "当前正式模型要求 nodes.cloud.resource_sharing=false；"
            "Cloud 使用固定每任务资源，不按同槽任务数均分。"
        )
    uav_deployment_altitude = float(
        get_config_value(config, "nodes.uav.deployment_altitude_m")
    )
    uav_channel_altitude = float(get_config_value(config, "nodes.uav.altitude_m"))
    if not math.isclose(
        uav_deployment_altitude,
        uav_channel_altitude,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "nodes.uav.deployment_altitude_m 必须与 nodes.uav.altitude_m 一致，"
            "以保持 UAV 部署与通信高度的单一合同。"
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
    if float(get_config_value(config, "nodes.cloud.cpu_cycles_per_s")) != float(
        get_config_value(config, "nodes.cloud.total_cpu_cycles_per_s")
    ):
        raise ValueError("Cloud 重复 CPU 字段必须与唯一总 CPU 容量一致。")
    if float(get_config_value(config, "nodes.cloud.uplink_rate_bps")) != float(
        get_config_value(config, "nodes.cloud.total_bandwidth_bps")
    ):
        raise ValueError("Cloud 重复 uplink 字段必须与唯一总带宽容量一致。")

    for key, status in config.get("parameter_status", {}).items():
        if status not in {
            "CONFIRMED_DATA",
            "TRAIN_ONLY_CALIBRATION",
            "EXPERIMENT_DESIGN",
            "ALGORITHM_DESIGN",
            "LITERATURE_DERIVED",
            "PROVISIONAL_IMPLEMENTATION",
            "TODO_LITERATURE_CALIBRATION",
        }:
            raise ValueError(f"未知参数状态 {status}：{key}")

    if require_artifacts:
        for key in (
            "paths.service_catalog",
            "paths.vehicle_roles",
            "paths.prediction_scenarios",
            "paths.sumo_net",
            "paths.node_positions",
            "paths.popularity_gru",
            "paths.tv_local_cache",
        ):
            _require_path_exists(config, key)
