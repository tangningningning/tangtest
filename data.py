"""论文数据层。

本模块是整个仿真的数据入口，负责把 ``config.yaml``、原始郑州 SUMO 场景、
请求工作负载和预测器输入整理成后续环境可以直接消费的结构。推荐阅读顺序为：

1. :func:`load_config` 解析配置并生成项目内绝对路径；
2. :func:`load_sumo_scene` 读取原始路网，必要时由 :func:`start_sumo` 启动场景；
3. :func:`generate_workload` 或 :func:`load_workload_npz` 生成/读取请求；
4. :func:`get_slot_data` 为环境提供单个决策时隙的数据切片。

本模块只处理“数据从哪里来、形状是什么、如何做最小合法性检查”。通信速率、
任务执行、缓存、matching、KKT、reward 和 PPO 不在此重复实现。SUMO 控制统一
通过官方 ``traci``/``libsumo`` API 完成；旧工程只能作为只读字段契约参考。
旧 GRU 只保留为隔离的离线 baseline 冻结推理接口，本文件不包含 GRU 训练流程；
正式需求预测使用 MobilityAwarePopularityPredictor。
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check is environment-specific
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
# 默认配置永远绑定当前新项目根目录，避免运行目录变化导致读取旧工程文件。
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# 未到达任务在 service type 数组中的哨兵值；真正到达的服务编号必须位于 [0,F)。
NO_TASK_SERVICE = -1


def _require_yaml() -> Any:
    if yaml is None:
        raise RuntimeError(
            "读取config.yaml需要PyYAML；请在adaptive_cache_mec环境中安装PyYAML。"
        ) from _YAML_IMPORT_ERROR
    return yaml


def _resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    """把配置中的路径解析为当前项目内的绝对路径。

    相对路径统一相对于 ``adaptive_cache_mec/`` 解析，而不是相对于调用进程的
    当前工作目录解析。这样从项目根目录、IDE 或测试入口启动时，SUMO 文件仍然
    指向同一份数据。该函数只解析路径，不创建、复制或修改任何文件。
    """

    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _as_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}必须是有限数值。")
    return result


def _as_positive(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if result <= 0:
        raise ValueError(f"{name}必须大于0。")
    return result


def _as_nonnegative(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if result < 0:
        raise ValueError(f"{name}不能小于0。")
    return result


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"配置节{name!r}必须是映射。")
    return value


def _dbm_to_w(value: Any, name: str) -> float:
    """把配置层的 dBm 总功率转换成内部使用的 W。"""

    dbm = _as_float(value, name)
    watts = 10.0 ** ((dbm - 30.0) / 10.0)
    return _as_positive(watts, f"{name}转换后的W")


def _db_power_gain_to_linear(value: Any, name: str) -> float:
    """把功率增益 dB 转换成线性功率增益；功率量使用除以10。"""

    gain_db = _as_float(value, name)
    linear_gain = 10.0 ** (gain_db / 10.0)
    return _as_positive(linear_gain, f"{name}转换后的线性增益")


def _normalize_engineering_units(config: dict[str, Any]) -> None:
    """在唯一配置入口把论文/工程单位转换成内部 SI/线性量。

    YAML 只保存人类可读的 Mbps、dBm 和 dB；转换结果写回同一份已加载配置，
    供 environment/solver 使用。这里是这些单位转换的唯一入口，物理计算层
    不再重复转换。
    """

    cloud = _section(config, "cloud")
    if "upload_rate_mbps" not in cloud:
        raise ValueError("cloud.upload_rate_mbps不能为空。")
    cloud["upload_rate_bps"] = _as_positive(
        cloud["upload_rate_mbps"], "cloud.upload_rate_mbps"
    ) * 1_000_000.0

    communication = _section(config, "communication")
    required_positive_parameters = (
        "tx_power_w",
        "reference_channel_gain",
        "terrestrial_path_loss_exponent",
    )
    for parameter_name in required_positive_parameters:
        if parameter_name not in communication:
            raise ValueError(f"communication.{parameter_name}不能为空。")
        communication[parameter_name] = _as_positive(
            communication[parameter_name], f"communication.{parameter_name}"
        )
    if "noise_power_dbm" not in communication:
        raise ValueError("communication.noise_power_dbm不能为空。")
    if "air_reference_gain_los_db" not in communication:
        raise ValueError("communication.air_reference_gain_los_db不能为空。")
    if "air_reference_gain_nlos_db" not in communication:
        raise ValueError("communication.air_reference_gain_nlos_db不能为空。")
    communication["noise_power_w"] = _dbm_to_w(
        communication["noise_power_dbm"], "communication.noise_power_dbm"
    )
    communication["air_reference_gain_los"] = _db_power_gain_to_linear(
        communication["air_reference_gain_los_db"],
        "communication.air_reference_gain_los_db",
    )
    communication["air_reference_gain_nlos"] = _db_power_gain_to_linear(
        communication["air_reference_gain_nlos_db"],
        "communication.air_reference_gain_nlos_db",
    )


def _vehicle_ids(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name}必须是车辆ID序列。")
    result = tuple(str(item) for item in value)
    if not result and not allow_empty or len(set(result)) != len(result):
        raise ValueError(f"{name}必须是非空且不重复的车辆ID序列。")
    return result


def stable_priority(seed: int, vehicle_id: str) -> int:
    """Return a reproducible, process-independent priority for one SUMO ID."""

    payload = f"{int(seed)}:{str(vehicle_id)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _sumo_route_files(configuration_file: str | os.PathLike[str]) -> tuple[Path, ...]:
    """从正式 ``.sumocfg`` 读取 route-files 的静态场景文件集合。

    这不是重写 SUMO 解析器：SUMO 仍由官方 TraCI/libsumo 负责运行和读取
    车辆状态；这里仅读取 SUMO 配置声明的 XML 文件名，用于建立完整候选全集。
    每个决策时隙的角色仍由当前活动车辆状态确定。
    """

    configuration_path = _resolve_path(configuration_file)
    if configuration_path is None or not configuration_path.is_file():
        raise FileNotFoundError(f"SUMO配置文件不存在：{configuration_path}")
    try:
        root = ET.parse(configuration_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"SUMO配置文件XML格式无效：{configuration_path}") from error
    route_element = root.find("./input/route-files")
    route_value = None if route_element is None else route_element.get("value")
    if route_value is None or not route_value.strip():
        raise ValueError(f"SUMO配置未声明input/route-files：{configuration_path}")

    route_files: list[Path] = []
    for item in route_value.split(","):
        route_name = item.strip()
        if not route_name:
            continue
        route_path = Path(os.path.expandvars(route_name)).expanduser()
        if not route_path.is_absolute():
            route_path = configuration_path.parent / route_path
        route_path = route_path.resolve()
        if not route_path.is_file():
            raise FileNotFoundError(f"SUMO route文件不存在：{route_path}")
        route_files.append(route_path)
    if not route_files:
        raise ValueError(f"SUMO配置的route-files为空：{configuration_path}")
    return tuple(route_files)


def _discover_vehicle_ids_from_routes(
    configuration_file: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Read the complete SUMO vehicle universe from every configured route file."""

    vehicle_ids: set[str] = set()
    for route_file in _sumo_route_files(configuration_file):
        try:
            for _, element in ET.iterparse(route_file, events=("end",)):
                tag = element.tag.rsplit("}", 1)[-1]
                if tag not in {"vehicle", "trip"}:
                    element.clear()
                    continue
                vehicle_id = str(element.get("id", "")).strip()
                if vehicle_id:
                    vehicle_ids.add(vehicle_id)
                element.clear()
        except ET.ParseError as error:
            raise ValueError(f"SUMO route文件XML格式无效：{route_file}") from error

    if not vehicle_ids:
        raise ValueError("正式SUMO场景没有可用车辆。")
    return tuple(sorted(vehicle_ids))


def _role_counts(config: Mapping[str, Any]) -> tuple[int, int]:
    roles = config.get("vehicle_roles")
    if not isinstance(roles, Mapping):
        raise ValueError("配置必须包含vehicle_roles映射。")
    task_count = int(roles.get("num_task_vehicles", -1))
    service_count = int(roles.get("num_service_vehicles", -1))
    if task_count < 0 or service_count < 0:
        raise ValueError("vehicle_roles中的车辆数量不能小于0。")
    return task_count, service_count


def _role_region(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    roles = config.get("vehicle_roles")
    if isinstance(roles, Mapping) and isinstance(roles.get("region"), Mapping):
        return roles["region"]
    uav = config.get("uav")
    if isinstance(uav, Mapping) and isinstance(uav.get("region"), Mapping):
        return uav["region"]
    return None


class VehicleRoleManager:
    """Manage task, service, and background roles using current SUMO states only."""

    def __init__(self, config: Mapping[str, Any], *, seed: int | None = None) -> None:
        self.config = config
        self.num_task_vehicles, self.num_service_vehicles = _role_counts(config)
        project = config.get("project")
        configured_seed = project.get("seed", 0) if isinstance(project, Mapping) else 0
        self.seed = int(configured_seed if seed is None else seed)
        roles = config.get("vehicle_roles")
        if not isinstance(roles, Mapping):
            raise ValueError("配置必须包含vehicle_roles映射。")
        self.score_alpha = float(roles.get("service_score_alpha", 0.5))
        self.v2v_range_m = float(roles.get("service_v2v_range_m", 300.0))
        self.min_service_stay_s = float(roles.get("min_service_stay_s", 0.0))
        self.stay_horizon_s = float(roles.get("service_stay_horizon_s", 60.0))
        self.exit_margin_m = float(roles.get("service_exit_margin_m", 0.0))
        if not 0.0 <= self.score_alpha <= 1.0:
            raise ValueError("vehicle_roles.service_score_alpha必须位于[0,1]。")
        for value, name in (
            (self.v2v_range_m, "service_v2v_range_m"),
            (self.min_service_stay_s, "min_service_stay_s"),
            (self.stay_horizon_s, "service_stay_horizon_s"),
            (self.exit_margin_m, "service_exit_margin_m"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"vehicle_roles.{name}必须是非负有限数。")
        if self.stay_horizon_s <= 0.0:
            raise ValueError("vehicle_roles.service_stay_horizon_s必须大于0。")
        self._region = _role_region(config)
        self._previous_roles: dict[str, tuple[str, ...]] = {
            "task_vehicle_ids": (),
            "service_vehicle_ids": (),
            "background_vehicle_ids": (),
        }

    @staticmethod
    def _role_ids(
        roles: Mapping[str, Any] | None,
        name: str,
    ) -> tuple[str, ...]:
        if not isinstance(roles, Mapping):
            return ()
        value = roles.get(name, roles.get(name.replace("_vehicle_ids", "_ids"), ()))
        if value is None:
            return ()
        return tuple(dict.fromkeys(str(item) for item in value))

    @staticmethod
    def _state_position(state: Mapping[str, Any]) -> np.ndarray | None:
        value = state.get("position_m", state.get("position"))
        if value is None:
            return None
        try:
            position = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            return None
        return position

    def _in_region(self, position: np.ndarray | None) -> bool:
        if position is None or self._region is None:
            return True
        try:
            return bool(
                float(self._region["x_min_m"]) <= position[0] <= float(self._region["x_max_m"])
                and float(self._region["y_min_m"]) <= position[1] <= float(self._region["y_max_m"])
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("vehicle_roles.region必须包含x/y方向的边界。")

    def _is_valid_state(self, state: Any) -> bool:
        if not isinstance(state, Mapping):
            return False
        for flag in ("valid", "active"):
            if flag in state and not bool(state[flag]):
                return False
        status = str(state.get("status", "")).strip().lower()
        if status in {"invalid", "departed", "arrived", "removed", "inactive"}:
            return False
        return self._in_region(self._state_position(state))

    @staticmethod
    def _velocity(state: Mapping[str, Any]) -> np.ndarray:
        try:
            speed_x = float(state.get("speed_x_mps", math.nan))
            speed_y = float(state.get("speed_y_mps", math.nan))
            if math.isfinite(speed_x) and math.isfinite(speed_y):
                return np.asarray((speed_x, speed_y), dtype=float)
            speed = float(state.get("speed_mps", 0.0))
            angle = float(state.get("angle_deg", 0.0))
            if not math.isfinite(speed) or speed < 0.0 or not math.isfinite(angle):
                return np.zeros(2, dtype=float)
            angle_rad = math.radians(angle)
            return np.asarray(
                (speed * math.sin(angle_rad), speed * math.cos(angle_rad)),
                dtype=float,
            )
        except (TypeError, ValueError):
            return np.zeros(2, dtype=float)

    def _estimated_stay_s(self, state: Mapping[str, Any]) -> float:
        position = self._state_position(state)
        if position is None or self._region is None:
            return math.inf
        velocity = self._velocity(state)
        boundaries = (
            (position[0], velocity[0], float(self._region["x_min_m"]), float(self._region["x_max_m"])),
            (position[1], velocity[1], float(self._region["y_min_m"]), float(self._region["y_max_m"])),
        )
        times: list[float] = []
        for coordinate, speed, lower, upper in boundaries:
            if speed > 1.0e-9:
                distance = upper - coordinate
                if distance <= self.exit_margin_m:
                    return 0.0
                times.append(max(0.0, distance / speed))
            elif speed < -1.0e-9:
                distance = coordinate - lower
                if distance <= self.exit_margin_m:
                    return 0.0
                times.append(max(0.0, distance / -speed))
        return min(times) if times else math.inf

    def _service_rejection_reason(self, state: Any) -> str | None:
        if not self._is_valid_state(state):
            return "vehicle_left_region_or_invalid"
        stay_s = self._estimated_stay_s(state)
        if stay_s < self.min_service_stay_s:
            return "stay_too_short"
        return None

    def _service_score(
        self,
        vehicle_id: str,
        state: Mapping[str, Any],
        task_ids: Sequence[str],
        states: Mapping[str, Mapping[str, Any]],
    ) -> float:
        stay_s = self._estimated_stay_s(state)
        stay_score = 1.0 if not math.isfinite(stay_s) else min(1.0, stay_s / self.stay_horizon_s)
        position = self._state_position(state)
        covered = 0
        if position is not None:
            for task_id in task_ids:
                task_position = self._state_position(states.get(task_id, {}))
                if task_position is not None and float(np.linalg.norm(position - task_position)) <= self.v2v_range_m:
                    covered += 1
        cover_score = covered / max(1, len(task_ids))
        return self.score_alpha * stay_score + (1.0 - self.score_alpha) * cover_score

    def _normalise_states(self, vehicle_states: Mapping[str, Any] | Sequence[str]) -> dict[str, Mapping[str, Any]]:
        if isinstance(vehicle_states, Mapping):
            return {
                str(vehicle_id): state if isinstance(state, Mapping) else {}
                for vehicle_id, state in vehicle_states.items()
            }
        return {str(vehicle_id): {} for vehicle_id in vehicle_states}

    def assign(
        self,
        vehicle_states: Mapping[str, Any] | Sequence[str],
        *,
        previous_roles: Mapping[str, Any] | None = None,
        slot: int | None = None,
    ) -> dict[str, Any]:
        """Assign roles for one slot; all selection inputs are current state fields."""

        states = self._normalise_states(vehicle_states)
        valid_ids = tuple(
            sorted(
                (vehicle_id for vehicle_id, state in states.items() if self._is_valid_state(state)),
                key=lambda vehicle_id: (stable_priority(self.seed, vehicle_id), vehicle_id),
            )
        )
        previous = self._previous_roles if previous_roles is None else previous_roles
        previous_tasks = self._role_ids(previous, "task_vehicle_ids")
        previous_services = self._role_ids(previous, "service_vehicle_ids")

        retained_services = [
            vehicle_id
            for vehicle_id in previous_services
            if vehicle_id in states
            and vehicle_id in valid_ids
            and self._service_rejection_reason(states[vehicle_id]) is None
        ][: self.num_service_vehicles]
        protected_services = set(retained_services)
        retained_tasks = [
            vehicle_id
            for vehicle_id in previous_tasks
            if vehicle_id in valid_ids and vehicle_id not in protected_services
        ][: self.num_task_vehicles]
        task_candidates = [
            vehicle_id
            for vehicle_id in valid_ids
            if vehicle_id not in protected_services and vehicle_id not in retained_tasks
        ]
        task_ids = tuple((retained_tasks + task_candidates)[: self.num_task_vehicles])

        service_candidates = [
            vehicle_id
            for vehicle_id in valid_ids
            if vehicle_id not in set(task_ids)
            and vehicle_id not in protected_services
            and self._service_rejection_reason(states[vehicle_id]) is None
        ]
        service_candidates.sort(
            key=lambda vehicle_id: (
                -self._service_score(vehicle_id, states[vehicle_id], task_ids, states),
                stable_priority(self.seed, vehicle_id),
                vehicle_id,
            )
        )
        service_ids = tuple(
            (retained_services + service_candidates)[: self.num_service_vehicles]
        )
        task_set = set(task_ids)
        service_set = set(service_ids)
        background_ids = tuple(vehicle_id for vehicle_id in valid_ids if vehicle_id not in task_set | service_set)

        previous_task_set = set(previous_tasks)
        previous_service_set = set(previous_services)
        task_leaves = [vehicle_id for vehicle_id in previous_tasks if vehicle_id not in task_set]
        task_joins = [vehicle_id for vehicle_id in task_ids if vehicle_id not in previous_task_set]
        events: list[dict[str, Any]] = []
        for vehicle_id in task_leaves:
            reason = "vehicle_left_region_or_invalid"
            if vehicle_id in states and self._is_valid_state(states[vehicle_id]):
                reason = "task_slot_released"
            events.append({
                "event": "task_vehicle_leave",
                "type": "task_vehicle_leave",
                "slot": slot,
                "vehicle_id": vehicle_id,
                "reason": reason,
            })
        for vehicle_id in task_joins:
            events.append({
                "event": "task_vehicle_join",
                "type": "task_vehicle_join",
                "slot": slot,
                "vehicle_id": vehicle_id,
                "reason": "initial_selection" if not previous_tasks else "task_replacement",
            })

        removed_services = [vehicle_id for vehicle_id in previous_services if vehicle_id not in service_set]
        added_services = [vehicle_id for vehicle_id in service_ids if vehicle_id not in previous_service_set]
        for index, old_id in enumerate(removed_services):
            new_id = added_services[index] if index < len(added_services) else None
            reason = "vehicle_left_region_or_invalid"
            if old_id in states:
                reason = self._service_rejection_reason(states[old_id]) or "service_vehicle_replaced"
            events.append({
                "event": "service_vehicle_replacement",
                "type": "service_vehicle_replacement",
                "slot": slot,
                "old_sv_id": old_id,
                "new_sv_id": new_id,
                "reason": reason,
            })
        for new_id in added_services[len(removed_services):]:
            events.append({
                "event": "service_vehicle_join",
                "type": "service_vehicle_join",
                "slot": slot,
                "old_sv_id": None,
                "new_sv_id": new_id,
                "reason": "service_vehicle_count_shortage",
            })

        result = {
            "task_vehicle_ids": task_ids,
            "service_vehicle_ids": service_ids,
            "background_vehicle_ids": background_ids,
            "role_change_events": tuple(events),
            "num_task_vehicles": len(task_ids),
            "num_service_vehicles": len(service_ids),
            "task_join_count": len(task_joins),
            "task_leave_count": len(task_leaves),
            "service_replace_count": len(removed_services),
        }
        self._previous_roles = {
            "task_vehicle_ids": task_ids,
            "service_vehicle_ids": service_ids,
            "background_vehicle_ids": background_ids,
        }
        return result

    update = assign
    select_roles = assign


def resolve_vehicle_roles(
    config: Mapping[str, Any],
    *,
    all_vehicle_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return deterministic ``(service_vehicle_ids, task_vehicle_ids)`` for a scene."""

    resolved = config.get("_resolved")
    if all_vehicle_ids is None and isinstance(resolved, Mapping):
        service_value = resolved.get("service_vehicle_ids")
        task_value = resolved.get("task_vehicle_ids")
        if service_value is not None and task_value is not None:
            task_count, service_count = _role_counts(config)
            resolved_service_ids = _vehicle_ids(
                service_value, "_resolved.service_vehicle_ids", allow_empty=True
            )
            resolved_task_ids = _vehicle_ids(
                task_value, "_resolved.task_vehicle_ids", allow_empty=True
            )
            if len(resolved_task_ids) == task_count and len(resolved_service_ids) == service_count:
                return resolved_service_ids, resolved_task_ids

    if all_vehicle_ids is None:
        resolved_all_ids = resolved.get("all_vehicle_ids") if isinstance(resolved, Mapping) else None
        if resolved_all_ids is not None:
            complete_ids = tuple(dict.fromkeys(str(item) for item in resolved_all_ids))
        else:
            sumo = _section(config, "sumo")
            configuration_file = sumo.get("configuration_file")
            if configuration_file is None:
                raise ValueError("未提供SUMO配置文件，无法推导场景车辆全集。")
            complete_ids = _discover_vehicle_ids_from_routes(configuration_file)
    else:
        complete_ids = tuple(dict.fromkeys(str(item) for item in all_vehicle_ids))
    if not complete_ids:
        raise ValueError("车辆全集不能为空。")
    assignment = VehicleRoleManager(config).assign(complete_ids, slot=0)
    return assignment["service_vehicle_ids"], assignment["task_vehicle_ids"]


def load_config(path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取唯一的项目配置、统一工程单位并进行最小结构检查。

    返回值保留 YAML 的原始层级，同时把 Mbps/dBm/dB 转换为内部使用的
    bit/s、W 和线性功率增益，并增加 ``config["_resolved"]``，供数据层和环境层
    使用已经解析好的项目根目录、SUMO 路网和车辆/服务数量。这里不主动生成
    随机数据，也不启动 SUMO，因此可以安全地用于参数审计。
    """

    yaml_module = _require_yaml()
    config_path = _resolve_path(path)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"config.yaml不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml_module.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.yaml顶层必须是映射。")
    _normalize_engineering_units(config)

    simulation = _section(config, "simulation")
    if _as_positive(simulation["decision_interval_s"], "simulation.decision_interval_s") <= 0:
        raise ValueError("决策间隔必须大于0。")
    slot_count = int(simulation["slot_count"])
    if slot_count <= 0:
        raise ValueError("simulation.slot_count必须是正整数。")

    task_vehicle_count, service_vehicle_count = _role_counts(config)

    services = _section(config, "services")
    service_count = int(services["count"])
    if service_count <= 0:
        raise ValueError("services.count必须是正整数。")

    sumo = _section(config, "sumo")
    if str(sumo.get("backend", "traci")).lower() not in {"traci", "libsumo"}:
        raise ValueError("sumo.backend只能是traci或libsumo。")
    network_file = _resolve_path(sumo.get("network_file"))
    if network_file is None:
        raise ValueError("sumo.network_file不能为空。")
    configuration_file = sumo.get("configuration_file")
    if configuration_file is None:
        raise ValueError("sumo.configuration_file不能为空，无法推导场景车辆集合。")
    mobility = config.get("mobility", {})
    if mobility is not None and not isinstance(mobility, Mapping):
        raise ValueError("mobility必须是映射。")
    if isinstance(mobility, Mapping):
        mobility_source = str(mobility.get("source", "sumo")).strip().lower()
        if mobility_source not in {"sumo", "trace"}:
            raise ValueError("mobility.source只能是sumo或trace。")
        if mobility_source == "trace" and not str(
            mobility.get("trace_root", "")
        ).strip():
            raise ValueError("mobility.source=trace时必须提供mobility.trace_root。")
    all_vehicle_ids = _discover_vehicle_ids_from_routes(configuration_file)
    assignment = VehicleRoleManager(config).assign(all_vehicle_ids, slot=0)
    service_ids = assignment["service_vehicle_ids"]
    task_ids = assignment["task_vehicle_ids"]
    if len(task_ids) < task_vehicle_count:
        raise ValueError(
            "SUMO场景可用车辆不足以满足vehicle_roles.num_task_vehicles。"
        )
    if len(service_ids) < service_vehicle_count:
        raise ValueError(
            "SUMO场景可用车辆不足以满足vehicle_roles.num_service_vehicles。"
        )

    config["_resolved"] = {
        "project_root": str(PROJECT_ROOT),
        "config_file": str(config_path),
        "network_file": str(network_file),
        "configuration_file": str(_resolve_path(configuration_file)),
        "all_vehicle_ids": all_vehicle_ids,
        "service_vehicle_ids": service_ids,
        "task_vehicle_ids": task_ids,
        "background_vehicle_ids": assignment["background_vehicle_ids"],
        "num_task_vehicles": task_vehicle_count,
        "num_service_vehicles": service_vehicle_count,
        "service_count": service_count,
    }
    return config


def _import_sumolib() -> Any:
    try:
        import sumolib
    except ImportError as exc:  # pragma: no cover - dependency check is environment-specific
        raise RuntimeError("读取SUMO路网需要sumolib。") from exc
    return sumolib


def _import_sumo_api(backend: str) -> Any:
    normalized = str(backend).strip().lower()
    if normalized == "libsumo":
        try:
            import libsumo as api
        except ImportError as exc:  # pragma: no cover - depends on SUMO installation
            raise RuntimeError("当前环境没有libsumo，请改用traci或安装匹配的SUMO。") from exc
        return api
    try:
        import traci as api
    except ImportError as exc:  # pragma: no cover - dependency check is environment-specific
        raise RuntimeError("当前环境没有traci，请安装官方SUMO Python接口。") from exc
    return api


def _network_bounds(network: Any) -> tuple[float, float, float, float]:
    boundary = network.getBoundary()
    if len(boundary) == 4 and not isinstance(boundary[0], (list, tuple, np.ndarray)):
        x_min, y_min, x_max, y_max = boundary
    elif len(boundary) == 2:
        (x_min, y_min), (x_max, y_max) = boundary
    else:
        raise ValueError("SUMO路网边界格式无效。")
    return (
        _as_float(x_min, "network.x_min"),
        _as_float(x_max, "network.x_max"),
        _as_float(y_min, "network.y_min"),
        _as_float(y_max, "network.y_max"),
    )


def load_sumo_scene(
    config: Mapping[str, Any] | None = None,
    *,
    configuration_file: str | os.PathLike[str] | None = None,
    start: bool = False,
) -> dict[str, Any]:
    """读取项目内原样迁入的郑州路网；只有显式要求时才启动 SUMO。

    ``network_file``、``configuration_file`` 和 route 文件都由当前项目配置的
    相对路径解析。路网读取使用 ``sumolib.net.readNet``，只返回路网对象和边界；
    ``start=True`` 时才会额外通过官方 TraCI/libsumo 接口启动正式场景。该函数
    不重建路网、不修改 XML，也不生成新的车辆轨迹。
    """

    loaded_config = dict(config) if config is not None else load_config()
    sumo = _section(loaded_config, "sumo")
    network_file = _resolve_path(sumo.get("network_file"))
    if network_file is None or not network_file.is_file():
        raise FileNotFoundError(f"SUMO路网文件不存在：{network_file}")

    sumolib_module = _import_sumolib()
    # 预测器需要在路口内看见 SUMO internal edge，才能把 A -> :junction -> B
    # 折叠成合法的 A -> B 道路转移；其余 SUMO 运行仍由 TraCI 正式场景负责。
    network = sumolib_module.net.readNet(str(network_file), withInternal=True)
    scene: dict[str, Any] = {
        "network": network,
        "network_file": str(network_file),
        "network_bounds": _network_bounds(network),
        "api": None,
    }

    if start:
        config_file = configuration_file or sumo.get("configuration_file")
        if config_file is None:
            raise ValueError(
                "当前配置只包含原始路网；启动SUMO时需要显式提供外部configuration_file。"
            )
        scene["api"] = start_sumo(
            config_file,
            backend=str(sumo.get("backend", "traci")),
            executable=sumo.get("executable"),
            seed=int(_section(loaded_config, "project").get("seed", 0)),
        )
    return scene


def _resolve_executable(executable: str | os.PathLike[str] | None, gui: bool) -> str:
    if executable:
        candidate = _resolve_path(executable)
        if candidate is not None and candidate.is_file():
            return str(candidate)
        located = shutil.which(str(executable))
        if located:
            return located
        raise FileNotFoundError(f"SUMO可执行文件不存在：{executable}")
    names = ("sumo-gui", "sumo") if gui else ("sumo", "sumo-gui")
    for name in names:
        located = shutil.which(name)
        if located:
            return located
    raise FileNotFoundError("未找到sumo或sumo-gui可执行文件。")


def start_sumo(
    configuration_file: str | os.PathLike[str],
    *,
    backend: str = "traci",
    executable: str | os.PathLike[str] | None = None,
    gui: bool = False,
    seed: int | None = None,
) -> Any:
    """使用官方 ``traci``/``libsumo`` 启动一个外部 SUMO 场景。

    ``configuration_file`` 是 SUMO 的 ``.sumocfg``，其中的 network、route 和
    view 相对引用由 SUMO 自己解析。这里仅负责选择后端、拼接启动命令和传递
    随机种子，不实现自定义仿真器。
    """

    config_path = _resolve_path(configuration_file)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"SUMO配置文件不存在：{config_path}")
    command = [
        _resolve_executable(executable, gui),
        "-c",
        str(config_path),
        "--message-log",
        os.devnull,
        # 正式实验默认关闭逐步与耗时统计输出；真实 SUMO warning/error 仍保留。
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--duration-log.statistics",
        "false",
    ]
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    api = _import_sumo_api(backend)
    api.start(command)
    return api


def reset(api: Any, configuration_file: str | os.PathLike[str]) -> Any:
    """通过官方 API 重新加载 SUMO 配置并返回同一个连接对象。"""

    config_path = _resolve_path(configuration_file)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"SUMO配置文件不存在：{config_path}")
    if api is None or not hasattr(api, "load"):
        raise ValueError("reset需要已经启动的traci/libsumo API对象。")
    api.load(
        [
            "-c",
            str(config_path),
            "--message-log",
            os.devnull,
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--duration-log.statistics",
            "false",
        ]
    )
    return api


def step(api: Any, target_time_s: float | None = None) -> Any:
    """推进 SUMO 到下一步或指定仿真时间。

    环境层以决策时隙时间调用本函数；如果传入目标时间，TraCI/libsumo 会负责
    内部推进中间的仿真步。数据层不对车辆运动做二次积分，避免和 SUMO 产生两
    套不一致的轨迹。
    """

    if api is None or not hasattr(api, "simulationStep"):
        raise ValueError("step需要已经启动的traci/libsumo API对象。")
    if target_time_s is None:
        return api.simulationStep()
    return api.simulationStep(float(target_time_s))


def close(api: Any) -> None:
    """关闭当前 TraCI/libsumo 连接；空连接可安全调用。"""

    if api is not None and hasattr(api, "close"):
        api.close()


def _vehicle_route_id(api: Any, vehicle_id: str) -> str | None:
    try:
        return str(api.vehicle.getRouteID(vehicle_id))
    except Exception:
        return None


def get_vehicle_states(
    api: Any,
    vehicle_ids: Sequence[str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    """读取论文需要的车辆 ID、位置、二维速度、方向、route 和仿真时间。

    返回字典的每个键对应一辆车，``position_m`` 为二维米制坐标，``speed_mps``
    为 SUMO 标量速度；``speed_x_mps``/``speed_y_mps`` 按 SUMO 导航角分解。
    默认只返回当前SUMO活动集合中的车辆；``strict=True`` 时，显式请求的离场
    车辆会抛出错误。角色管理器始终使用默认的当前集合，不读取未来轨迹。
    """

    if api is None:
        raise ValueError("get_vehicle_states需要已经启动的traci/libsumo API对象。")
    active_vehicle_ids = tuple(str(item) for item in api.vehicle.getIDList())
    active_vehicle_set = set(active_vehicle_ids)
    ids = (
        active_vehicle_ids
        if vehicle_ids is None
        else tuple(str(item) for item in vehicle_ids)
    )
    missing = tuple(vehicle_id for vehicle_id in ids if vehicle_id not in active_vehicle_set)
    if missing and strict:
        raise RuntimeError(
            "请求读取的车辆不在当前SUMO活动集合中："
            f"{missing}"
        )
    ids = tuple(vehicle_id for vehicle_id in ids if vehicle_id in active_vehicle_set)
    simulation_time = float(api.simulation.getTime())
    states: dict[str, dict[str, Any]] = {}
    for vehicle_id in ids:
        position = np.asarray(api.vehicle.getPosition(vehicle_id), dtype=np.float64)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"车辆{vehicle_id}的位置无效。")
        speed = float(api.vehicle.getSpeed(vehicle_id))
        try:
            angle = float(api.vehicle.getAngle(vehicle_id))
            angle_rad = math.radians(angle)
            speed_x = speed * math.sin(angle_rad)
            speed_y = speed * math.cos(angle_rad)
        except Exception:
            angle = math.nan
            speed_x = math.nan
            speed_y = math.nan
        try:
            lane_id = str(api.vehicle.getLaneID(vehicle_id))
        except Exception:
            lane_id = None
        try:
            edge_id = str(api.vehicle.getRoadID(vehicle_id))
        except Exception:
            edge_id = None
        try:
            lane_position = float(api.vehicle.getLanePosition(vehicle_id))
            if not math.isfinite(lane_position):
                lane_position = math.nan
        except Exception:
            lane_position = math.nan
        try:
            acceleration = float(api.vehicle.getAcceleration(vehicle_id))
            if not math.isfinite(acceleration):
                acceleration = 0.0
        except Exception:
            acceleration = 0.0
        try:
            allowed_speed = float(api.vehicle.getAllowedSpeed(vehicle_id))
            if not math.isfinite(allowed_speed) or allowed_speed < 0.0:
                allowed_speed = math.nan
        except Exception:
            allowed_speed = math.nan
        try:
            lane_index = int(api.vehicle.getLaneIndex(vehicle_id))
        except Exception:
            lane_index = None
        states[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "position_m": position,
            "speed_mps": speed,
            "speed_x_mps": speed_x,
            "speed_y_mps": speed_y,
            "angle_deg": angle,
            "lane_id": lane_id,
            "lane_index": lane_index,
            "lane_position_m": lane_position,
            "edge_id": edge_id,
            "acceleration_mps2": acceleration,
            "allowed_speed_mps": allowed_speed,
            "route_id": _vehicle_route_id(api, vehicle_id),
            "active": True,
            "valid": True,
            "simulation_time_s": simulation_time,
        }
    return states


class RoadTransitionModel:
    """基于已观测道路序列的一阶合法 edge 转移模型。

    该模型只从连续时刻的 ``getRoadID`` 观测更新计数，不读取 ``getRoute``、
    ``.rou.xml`` 中的未来路线或离线未来轨迹。SUMO internal edge 只作为路口
    过渡被跳过，因此统计的是可解释的普通道路 A -> B 转移。
    """

    def __init__(self, network: Any) -> None:
        if network is None or not hasattr(network, "getEdge"):
            raise ValueError("RoadTransitionModel需要sumolib道路网络对象")
        self.network = network
        self.transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._last_normal_edge_by_vehicle: dict[str, str] = {}

    def _edge(self, edge_id: Any) -> Any | None:
        if edge_id is None:
            return None
        try:
            return self.network.getEdge(str(edge_id))
        except Exception:
            return None

    @staticmethod
    def _is_internal_edge(edge: Any | None, edge_id: str | None = None) -> bool:
        if edge is not None:
            try:
                return str(edge.getFunction()).lower() == "internal"
            except Exception:
                pass
        # 只有网络对象无法解析时才作防御性回退；正常路径优先使用 SUMO function。
        return bool(edge_id and str(edge_id).startswith(":"))

    def is_internal_edge(self, edge_id: Any) -> bool:
        edge_name = None if edge_id is None else str(edge_id)
        return self._is_internal_edge(self._edge(edge_name), edge_name)

    def _normal_outgoing_edges(self, edge: Any | None) -> tuple[Any, ...]:
        """递归跳过 internal edge，得到当前 edge 的合法普通出口。"""

        if edge is None:
            return ()
        result: dict[str, Any] = {}
        pending = list(getattr(edge, "getOutgoing", lambda: {})().keys())
        visited: set[str] = set()
        while pending:
            candidate = pending.pop(0)
            try:
                candidate_id = str(candidate.getID())
            except Exception:
                continue
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            if self._is_internal_edge(candidate, candidate_id):
                pending.extend(
                    list(getattr(candidate, "getOutgoing", lambda: {})().keys())
                )
            else:
                result[candidate_id] = candidate
        return tuple(result[key] for key in sorted(result))

    def legal_outgoing_edge_ids(self, edge_id: Any) -> tuple[str, ...]:
        edge_name = None if edge_id is None else str(edge_id)
        return tuple(
            str(edge.getID()) for edge in self._normal_outgoing_edges(self._edge(edge_name))
        )

    def last_normal_edge(self, vehicle_id: str) -> str | None:
        return self._last_normal_edge_by_vehicle.get(str(vehicle_id))

    def observe_state(self, vehicle_id: str, state: Mapping[str, Any]) -> None:
        """只用当前/历史观测更新一次普通道路转移计数。"""

        if not bool(state.get("active", True)) or not bool(state.get("valid", True)):
            return
        edge_value = state.get("edge_id")
        if edge_value is None:
            return
        edge_name = str(edge_value)
        if self.is_internal_edge(edge_name):
            # 不把 :junction_x 计入道路序列，保留上一个普通道路，等待下一条普通道路。
            return
        previous = self._last_normal_edge_by_vehicle.get(str(vehicle_id))
        if previous is not None and previous != edge_name:
            self.transition_counts[previous][edge_name] += 1
        self._last_normal_edge_by_vehicle[str(vehicle_id)] = edge_name

    def release_vehicle(self, vehicle_id: str) -> None:
        """车辆离场后释放在线上下文；全局 transition_counts 保留。"""

        self._last_normal_edge_by_vehicle.pop(str(vehicle_id), None)

    def next_edge_distribution(
        self,
        edge_id: Any,
        *,
        legal_edge_ids: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """返回只包含合法出口的一阶 P(e_next|e_current)。"""

        current = None if edge_id is None else str(edge_id)
        legal = tuple(
            sorted(
                set(str(item) for item in (legal_edge_ids or self.legal_outgoing_edge_ids(current)))
            )
        )
        if not legal:
            return {}
        counts = self.transition_counts.get(current, Counter())
        filtered = {candidate: float(counts[candidate]) for candidate in legal if counts[candidate] > 0}
        total = float(sum(filtered.values()))
        if total <= 0.0:
            # 没有历史时只在 SUMO 合法出口上均匀回退，绝不凭空生成非法道路。
            probability = 1.0 / len(legal)
            return {candidate: probability for candidate in legal}
        return {candidate: filtered.get(candidate, 0.0) / total for candidate in legal}


class RoadConstrainedVehiclePredictor:
    """SUMO lane 内有界运动 + 一阶道路转移的单路径预测器。

    第一版只保留概率最高的一条预测路线，分支概率固定为 1.0。车辆历史保存
    ``vehicle_id/slot/edge/lane/s/v/a`` 等当前观测；预测过程不调用未来路线接口。
    """

    _HISTORY_FIELDS = (
        "vehicle_id",
        "slot",
        "time_s",
        "edge_id",
        "lane_id",
        "lane_position_m",
        "position_m",
        "speed_mps",
        "acceleration_mps2",
    )

    def __init__(
        self,
        network: Any,
        *,
        dt_s: float,
        horizon_slots: int = 3,
        history_length_slots: int = 10,
        transition_model: RoadTransitionModel | None = None,
    ) -> None:
        self.network = network
        self.dt_s = _as_positive(dt_s, "trajectory_prediction.dt_s")
        self.horizon_slots = int(horizon_slots)
        self.history_length_slots = int(history_length_slots)
        if self.horizon_slots <= 0:
            raise ValueError("trajectory_prediction.horizon_slots必须为正整数")
        if self.history_length_slots <= 0:
            raise ValueError("trajectory_prediction.history_length_slots必须为正整数")
        self.transition_model = transition_model or RoadTransitionModel(network)
        self.history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.history_length_slots)
        )
        self.observed_normal_edge_sequence: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=self.history_length_slots)
        )
        self._observation_counter = 0

    @staticmethod
    def _finite_or(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        return numeric if math.isfinite(numeric) else float(default)

    def observe(
        self,
        states: Mapping[str, Mapping[str, Any]],
        *,
        slot: int | None = None,
    ) -> None:
        """登记当前观测；缺失车辆视为离场，但不删除全局转移统计。"""

        if not isinstance(states, Mapping):
            raise TypeError("RoadConstrainedVehiclePredictor.observe需要车辆状态映射")
        current_ids = {str(vehicle_id) for vehicle_id in states}
        previous_ids = set(self.history)
        for vehicle_id in sorted(previous_ids - current_ids):
            self.transition_model.release_vehicle(vehicle_id)
            self.history.pop(vehicle_id, None)
            self.observed_normal_edge_sequence.pop(vehicle_id, None)
        observation_slot = self._observation_counter if slot is None else int(slot)
        for raw_vehicle_id, raw_state in states.items():
            if not isinstance(raw_state, Mapping):
                continue
            vehicle_id = str(raw_vehicle_id)
            # 这里写入的全部字段来自当前/已经发生的观测，不读取未来 route 或轨迹。
            record = {
                "vehicle_id": vehicle_id,
                "slot": observation_slot,
                "time_s": self._finite_or(raw_state.get("simulation_time_s"), math.nan),
                "edge_id": raw_state.get("edge_id"),
                "lane_id": raw_state.get("lane_id"),
                "lane_position_m": self._finite_or(raw_state.get("lane_position_m"), math.nan),
                "position_m": np.asarray(raw_state.get("position_m", (math.nan, math.nan)), dtype=float).copy(),
                "speed_mps": self._finite_or(raw_state.get("speed_mps"), 0.0),
                "acceleration_mps2": self._finite_or(raw_state.get("acceleration_mps2"), 0.0),
            }
            self.history[vehicle_id].append(record)
            self.transition_model.observe_state(vehicle_id, raw_state)
            edge_id = raw_state.get("edge_id")
            if edge_id is not None and not self.transition_model.is_internal_edge(edge_id):
                sequence = self.observed_normal_edge_sequence[vehicle_id]
                if not sequence or sequence[-1] != str(edge_id):
                    sequence.append(str(edge_id))
        self._observation_counter = max(self._observation_counter + 1, observation_slot + 1)

    def _edge(self, edge_id: Any) -> Any | None:
        return self.transition_model._edge(edge_id)

    def _lane(self, edge: Any, lane_id: Any) -> Any | None:
        if edge is None:
            return None
        if lane_id is not None:
            try:
                lane = self.network.getLane(str(lane_id))
                if lane.getEdge().getID() == edge.getID():
                    return lane
            except Exception:
                pass
        lanes = list(getattr(edge, "getLanes", lambda: ())())
        return lanes[0] if lanes else None

    @staticmethod
    def _lane_offset_position(lane: Any, lane_position: float, fallback: np.ndarray) -> np.ndarray:
        try:
            from sumolib.geomhelper import positionAtShapeOffset

            point = positionAtShapeOffset(lane.getShape(), float(lane_position))
            result = np.asarray(point[:2], dtype=float)
            if result.shape == (2,) and np.all(np.isfinite(result)):
                return result
        except Exception:
            pass
        # 正式 SUMO lane 都有 shape；异常时保持最后一个合法 lane 坐标，禁止直线外推。
        return np.asarray(fallback, dtype=float).copy()

    def _choose_next_edge(
        self,
        vehicle_id: str,
        current_edge_id: str,
    ) -> tuple[str | None, float, dict[str, float]]:
        legal = self.transition_model.legal_outgoing_edge_ids(current_edge_id)
        if not legal:
            return None, 0.0, {}
        context = current_edge_id
        if self.transition_model.is_internal_edge(current_edge_id):
            context = self.transition_model.last_normal_edge(vehicle_id) or current_edge_id
        distribution = self.transition_model.next_edge_distribution(
            context,
            legal_edge_ids=legal,
        )
        if not distribution:
            return None, 0.0, {}
        selected, probability = sorted(
            distribution.items(), key=lambda item: (-item[1], item[0])
        )[0]
        return str(selected), float(probability), distribution

    def predict_vehicle(
        self,
        vehicle_id: str,
        state: Mapping[str, Any],
        *,
        origin_slot: int = 0,
        horizon_slots: int | None = None,
    ) -> dict[str, Any]:
        """从当前 lane 状态滚动预测；路口处只选择最高概率合法出口。"""

        if not isinstance(state, Mapping):
            raise TypeError("state必须是车辆当前状态映射")
        vehicle_name = str(vehicle_id)
        horizon = self.horizon_slots if horizon_slots is None else int(horizon_slots)
        if horizon <= 0:
            raise ValueError("horizon_slots必须为正整数")
        origin_position = np.asarray(state.get("position_m", (math.nan, math.nan)), dtype=float)
        if origin_position.shape != (2,) or not np.all(np.isfinite(origin_position)):
            raise ValueError(f"车辆{vehicle_name}当前position_m无效")
        edge_value = state.get("edge_id")
        edge = self._edge(edge_value)
        lane = self._lane(edge, state.get("lane_id"))
        # 正常 SUMO 状态一定可解析到 edge/lane；异常状态保留当前位置，保证接口无 NaN。
        if edge is None or lane is None:
            positions = np.repeat(origin_position[np.newaxis, :], horizon, axis=0)
            branch = {
                "probability": 1.0,
                "positions_m": positions,
                "edge_ids": tuple(),
                "future_edge_ids": tuple(None for _ in range(horizon)),
                "lane_ids": tuple(None for _ in range(horizon)),
                "lane_positions_m": np.zeros(horizon, dtype=float),
                "chosen_transition_probability": 1.0,
            }
            return {
                "vehicle_id": vehicle_name,
                "prediction_origin_slot": int(origin_slot),
                "origin_slot": int(origin_slot),
                "future_slots": tuple(int(origin_slot) + index + 1 for index in range(horizon)),
                "origin_position_m": origin_position.copy(),
                "trajectory_branches": [branch],
                "predicted_positions_m": positions.copy(),
                "predicted_edge_ids": branch["future_edge_ids"],
                "next_edge_distribution": {},
                "censored": True,
            }

        current_edge_id = str(edge.getID())
        lane_position = self._finite_or(state.get("lane_position_m"), 0.0)
        lane_length = max(0.0, self._finite_or(lane.getLength(), 0.0))
        lane_position = float(np.clip(lane_position, 0.0, lane_length))
        speed = max(0.0, self._finite_or(state.get("speed_mps"), 0.0))
        acceleration = self._finite_or(state.get("acceleration_mps2"), 0.0)
        fallback_position = origin_position.copy()
        positions: list[np.ndarray] = []
        future_edge_ids: list[str] = []
        future_lane_ids: list[str | None] = []
        future_lane_positions: list[float] = []
        path_edge_ids: list[str] = [current_edge_id]
        chosen_transition_probability = 1.0
        initial_distribution: dict[str, float] = {}

        for _ in range(horizon):
            allowed_speed = self._finite_or(state.get("allowed_speed_mps"), math.nan)
            if not math.isfinite(allowed_speed) or allowed_speed < 0.0:
                allowed_speed = max(0.0, self._finite_or(lane.getSpeed(), speed))
            next_speed = float(np.clip(speed + acceleration * self.dt_s, 0.0, allowed_speed))
            distance = max(0.0, 0.5 * (speed + next_speed) * self.dt_s)
            while distance > 1.0e-9:
                remaining = max(0.0, lane_length - lane_position)
                if distance <= remaining + 1.0e-9:
                    lane_position += distance
                    distance = 0.0
                    speed = next_speed
                    break
                distance = max(0.0, distance - remaining)
                lane_position = lane_length
                next_edge_id, transition_probability, distribution = self._choose_next_edge(
                    vehicle_name, current_edge_id
                )
                if not initial_distribution:
                    initial_distribution = dict(distribution)
                if next_edge_id is None:
                    speed = 0.0
                    distance = 0.0
                    break
                chosen_transition_probability *= transition_probability
                next_edge = self._edge(next_edge_id)
                next_lane = self._lane(next_edge, None)
                if next_edge is None or next_lane is None:
                    speed = 0.0
                    distance = 0.0
                    break
                current_edge_id = str(next_edge.getID())
                edge = next_edge
                lane = next_lane
                lane_length = max(0.0, self._finite_or(lane.getLength(), 0.0))
                lane_position = 0.0
                path_edge_ids.append(current_edge_id)
                # 进入新 lane 后仍沿 lane 几何运动，速度在下一 slot 再按新限速截断。
                speed = next_speed
            point = self._lane_offset_position(lane, lane_position, fallback_position)
            fallback_position = point.copy()
            positions.append(point)
            future_edge_ids.append(current_edge_id)
            future_lane_ids.append(str(lane.getID()) if lane is not None else None)
            future_lane_positions.append(float(lane_position))

        position_array = np.asarray(positions, dtype=float)
        if position_array.shape != (horizon, 2) or not np.all(np.isfinite(position_array)):
            raise ValueError(f"车辆{vehicle_name}道路预测产生无效坐标")
        branch = {
            "probability": 1.0,
            "positions_m": position_array.copy(),
            "edge_ids": tuple(dict.fromkeys(path_edge_ids)),
            "future_edge_ids": tuple(future_edge_ids),
            "lane_ids": tuple(future_lane_ids),
            "lane_positions_m": np.asarray(future_lane_positions, dtype=float),
            "chosen_transition_probability": float(chosen_transition_probability),
        }
        return {
            "vehicle_id": vehicle_name,
            "prediction_origin_slot": int(origin_slot),
            "origin_slot": int(origin_slot),
            "future_slots": tuple(int(origin_slot) + index + 1 for index in range(horizon)),
            "origin_position_m": origin_position.copy(),
            "trajectory_branches": [branch],
            "predicted_positions_m": position_array.copy(),
            "predicted_edge_ids": tuple(future_edge_ids),
            "next_edge_distribution": dict(initial_distribution),
            "censored": False,
        }

    def predict(
        self,
        states: Mapping[str, Mapping[str, Any]],
        *,
        origin_slot: int = 0,
        update_history: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """批量预测；默认先登记当前观测，保证预测严格因果。"""

        if update_history:
            self.observe(states, slot=origin_slot)
        return {
            str(vehicle_id): self.predict_vehicle(
                str(vehicle_id), state, origin_slot=origin_slot
            )
            for vehicle_id, state in states.items()
            if isinstance(state, Mapping)
            and bool(state.get("active", True))
            and bool(state.get("valid", True))
        }

    @staticmethod
    def positions_matrix(
        predictions: Mapping[str, Mapping[str, Any]],
        vehicle_ids: Sequence[str],
    ) -> np.ndarray:
        result = np.asarray(
            [np.asarray(predictions[str(vehicle_id)]["predicted_positions_m"], dtype=float)[0]
             for vehicle_id in vehicle_ids],
            dtype=float,
        )
        if result.shape != (len(vehicle_ids), 2) or not np.all(np.isfinite(result)):
            raise ValueError("道路预测位置矩阵必须是有限的(V,2)数组")
        return result

    def predict_next_positions(
        self,
        states: Mapping[str, Mapping[str, Any]],
        vehicle_ids: Sequence[str] | None = None,
        *,
        origin_slot: int = 0,
        update_history: bool = True,
    ) -> np.ndarray:
        predictions = self.predict(
            states,
            origin_slot=origin_slot,
            update_history=update_history,
        )
        ids = tuple(str(item) for item in (vehicle_ids or tuple(predictions)))
        return self.positions_matrix(predictions, ids)

    def predict_residence_time(
        self,
        prediction: Mapping[str, Any],
        *,
        target_position: Sequence[float] | np.ndarray | None = None,
        node_position: Sequence[float] | np.ndarray | None = None,
        radius_m: float | None = None,
        region: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """计算覆盖窗口内的驻留下界；窗口末仍在内时标记 censored。"""

        center_value = target_position if target_position is not None else node_position
        if center_value is None and region is None:
            raise ValueError("驻留时间需要target_position/node_position或region")
        if center_value is not None:
            center = np.asarray(center_value, dtype=float)
            if center.shape != (2,) or not np.all(np.isfinite(center)):
                raise ValueError("target_position必须是有限的二维坐标")
            radius = 0.0 if radius_m is None else _as_nonnegative(radius_m, "radius_m")
        else:
            center = None
            radius = 0.0

        def inside(point: np.ndarray) -> bool:
            if center is not None:
                return bool(np.linalg.norm(point - center) <= radius + 1.0e-9)
            assert region is not None
            return bool(
                float(region["x_min_m"]) <= point[0] <= float(region["x_max_m"])
                and float(region["y_min_m"]) <= point[1] <= float(region["y_max_m"])
            )

        origin = np.asarray(prediction["origin_position_m"], dtype=float)
        branch = prediction["trajectory_branches"][0]
        positions = np.asarray(branch["positions_m"], dtype=float)
        coverage = np.asarray([inside(point) for point in np.vstack((origin, positions))], dtype=bool)
        if not coverage[0]:
            stay = 0.0
            censored = False
        else:
            outside = np.flatnonzero(~coverage[1:])
            if outside.size == 0:
                stay = float(positions.shape[0] * self.dt_s)
                censored = True
            else:
                stay = float((int(outside[0]) + 1) * self.dt_s)
                censored = False
        return {
            "residence_time_s": stay,
            "lower_bound_s": stay,
            "censored": censored,
            "coverage": coverage[1:].astype(float),
        }

    def predict_coverage(
        self,
        prediction: Mapping[str, Any],
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        coverage_radius_m: float,
    ) -> np.ndarray:
        """输出通用节点覆盖指标 I[v,m,t+h]；第一版为 0/1。"""

        nodes = np.asarray(node_positions, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1:] != (2,) or not np.all(np.isfinite(nodes)):
            raise ValueError("node_positions必须是有限的(M,2)数组")
        radius = _as_nonnegative(coverage_radius_m, "coverage_radius_m")
        branch_values: list[np.ndarray] = []
        for branch in prediction["trajectory_branches"]:
            positions = np.asarray(branch["positions_m"], dtype=float)
            distances = np.linalg.norm(positions[:, np.newaxis, :] - nodes[np.newaxis, :, :], axis=2)
            branch_values.append((distances <= radius + 1.0e-9).astype(float))
        if not branch_values:
            return np.zeros((0, nodes.shape[0]), dtype=float)
        probabilities = np.asarray(
            [float(branch["probability"]) for branch in prediction["trajectory_branches"]],
            dtype=float,
        )
        probabilities /= float(probabilities.sum())
        # 内部按 (horizon, node) 计算，接口整理为缓存模块更直观的 (node, horizon)。
        return np.tensordot(probabilities, np.asarray(branch_values), axes=(0, 0)).T

    def predict_batch_coverage(
        self,
        predictions: Mapping[str, Mapping[str, Any]],
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        coverage_radius_m: float,
    ) -> dict[str, np.ndarray]:
        return {
            str(vehicle_id): self.predict_coverage(prediction, node_positions, coverage_radius_m)
            for vehicle_id, prediction in predictions.items()
        }


class MobilityAwarePopularityPredictor:
    """用车辆历史服务偏好和未来覆盖估计节点服务需求。

    ``predict``/``predict_node_demand`` 只读取上一次 ``update_requests`` 之前的
    请求。正式环境在当前 slot 的预测完成后才调用 ``update_requests``，因此不会
    把 request(t) 泄漏到 prediction(t)。历史量采用单一 EWMA 衰减和一个全局先验
    平滑参数，不引入用户画像、区域模型或神经网络。
    """

    def __init__(
        self,
        service_count: int,
        *,
        history_decay: float = 0.8,
        preference_smoothing: float = 1.0,
        horizon_slots: int = 3,
    ) -> None:
        self.service_count = int(service_count)
        if self.service_count <= 0:
            raise ValueError("service_count必须为正整数")
        self.history_decay = _as_float(history_decay, "popularity_prediction.history_decay")
        if not 0.0 <= self.history_decay < 1.0:
            raise ValueError("popularity_prediction.history_decay必须位于[0,1)")
        self.preference_smoothing = _as_positive(
            preference_smoothing,
            "popularity_prediction.preference_smoothing",
        )
        self.horizon_slots = int(horizon_slots)
        if self.horizon_slots <= 0:
            raise ValueError("horizon_slots必须为正整数")
        self._global_ewma = np.full(
            self.service_count,
            1.0 / self.service_count,
            dtype=np.float64,
        )
        self._vehicle_ewma: dict[str, np.ndarray] = {}
        self._last_observed_slot = -1
        self._observed_request_slots: list[int] = []

    @property
    def last_observed_slot(self) -> int:
        return self._last_observed_slot

    @property
    def observed_request_slots(self) -> tuple[int, ...]:
        return tuple(self._observed_request_slots)

    @staticmethod
    def _normalise(values: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != fallback.shape or not np.all(np.isfinite(array)):
            return np.asarray(fallback, dtype=np.float64).copy()
        array = np.maximum(array, 0.0)
        total = float(array.sum())
        if total <= 0.0:
            return np.asarray(fallback, dtype=np.float64).copy()
        return array / total

    @property
    def global_preference(self) -> np.ndarray:
        return self._normalise(
            self._global_ewma,
            fallback=np.full(self.service_count, 1.0 / self.service_count),
        )

    def vehicle_preference(self, vehicle_id: str) -> np.ndarray:
        """用 EWMA 历史和全局先验做一次平滑，避免早期请求导致极端概率。"""

        history = self._vehicle_ewma.get(str(vehicle_id))
        global_preference = self.global_preference
        if history is None:
            return global_preference.copy()
        effective_count = float(np.maximum(history, 0.0).sum())
        if effective_count <= 0.0:
            return global_preference.copy()
        smoothed = (
            np.maximum(history, 0.0)
            + self.preference_smoothing * global_preference
        ) / (effective_count + self.preference_smoothing)
        return self._normalise(smoothed, fallback=global_preference)

    def _check_prediction_slot(self, origin_slot: int) -> int:
        slot = int(origin_slot)
        if slot <= self._last_observed_slot:
            raise ValueError(
                "预测slot必须晚于最近已登记请求slot；请先预测，再登记当前请求"
            )
        return slot

    def update_requests(
        self,
        slot: int,
        task_vehicle_ids: Sequence[str],
        task_arrivals: Sequence[bool] | np.ndarray,
        task_service_types: Sequence[int] | np.ndarray,
    ) -> None:
        """登记一个已经结束预测的请求 slot；该函数不能提前读取未来 slot。"""

        slot_index = int(slot)
        if slot_index <= self._last_observed_slot:
            raise ValueError("请求历史slot必须严格递增")
        vehicle_ids = tuple(str(item) for item in task_vehicle_ids)
        arrivals = np.asarray(task_arrivals, dtype=bool)
        service_ids = np.asarray(task_service_types, dtype=np.int64)
        if arrivals.shape != (len(vehicle_ids),) or service_ids.shape != arrivals.shape:
            raise ValueError("任务车辆、到达标记和服务ID形状必须一致")
        slot_counts = np.zeros(self.service_count, dtype=np.float64)
        decay = self.history_decay
        for index, vehicle_id in enumerate(vehicle_ids):
            if not bool(arrivals[index]):
                continue
            service_id = int(service_ids[index])
            if service_id < 0 or service_id >= self.service_count:
                continue
            one_hot = np.zeros(self.service_count, dtype=np.float64)
            one_hot[service_id] = 1.0
            previous = self._vehicle_ewma.get(
                vehicle_id,
                np.zeros(self.service_count, dtype=np.float64),
            )
            self._vehicle_ewma[vehicle_id] = decay * previous + (1.0 - decay) * one_hot
            slot_counts[service_id] += 1.0
        if float(slot_counts.sum()) > 0.0:
            # 全局 EWMA 只在有请求的 slot 更新；空 slot 不应把已有偏好人为衰减为零。
            observed_distribution = slot_counts / float(slot_counts.sum())
            self._global_ewma = (
                decay * self._global_ewma
                + (1.0 - decay) * observed_distribution
            )
        self._last_observed_slot = slot_index
        self._observed_request_slots.append(slot_index)

    def predict_global(self, origin_slot: int) -> np.ndarray:
        """返回截至 origin_slot-1 的全局历史先验。"""

        self._check_prediction_slot(origin_slot)
        return self.global_preference.copy()

    @staticmethod
    def _positions_from_prediction(value: Any, horizon: int) -> np.ndarray | None:
        if isinstance(value, Mapping):
            value = value.get("predicted_positions_m")
        if value is None:
            return None
        try:
            positions = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if positions.shape == (2,):
            positions = np.repeat(positions[np.newaxis, :], horizon, axis=0)
        if positions.shape != (horizon, 2) or not np.all(np.isfinite(positions)):
            return None
        return positions

    @staticmethod
    def _node_positions(node: Mapping[str, Any], horizon: int) -> np.ndarray:
        raw = node.get("positions_m", node.get("position_m"))
        positions = MobilityAwarePopularityPredictor._positions_from_prediction(raw, horizon)
        if positions is None:
            raise ValueError("节点positions_m必须是(2,)或(H,2)有限坐标")
        return positions

    @staticmethod
    def _node_radius(
        node: Mapping[str, Any],
        coverage_radius_m: float | Mapping[str, float],
    ) -> float:
        if "coverage_radius_m" in node:
            value = node["coverage_radius_m"]
        elif isinstance(coverage_radius_m, Mapping):
            node_type = str(node.get("node_type", "default"))
            value = coverage_radius_m.get(
                node_type,
                coverage_radius_m.get("default", 0.0),
            )
        else:
            value = coverage_radius_m
        return _as_nonnegative(value, "coverage_radius_m")

    def predict_node_demand(
        self,
        *,
        origin_slot: int,
        task_vehicle_ids: Sequence[str],
        vehicle_predictions: Mapping[str, Any] | None,
        nodes: Sequence[Mapping[str, Any]],
        coverage_radius_m: float | Mapping[str, float],
        horizon_slots: int | None = None,
    ) -> dict[str, Any]:
        """把车辆偏好与未来覆盖相乘，返回节点×服务需求和流行度。"""

        slot = self._check_prediction_slot(origin_slot)
        horizon = self.horizon_slots if horizon_slots is None else int(horizon_slots)
        if horizon <= 0:
            raise ValueError("horizon_slots必须为正整数")
        vehicle_ids = tuple(str(item) for item in task_vehicle_ids)
        predictions = vehicle_predictions or {}
        node_list = [dict(node) for node in nodes]
        node_ids = tuple(str(node.get("node_id")) for node in node_list)
        if any(node_id in {"None", ""} for node_id in node_ids):
            raise ValueError("每个缓存节点必须提供node_id")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("cache-capable节点ID不能重复")
        service_ids = tuple(range(self.service_count))
        preferences = {
            vehicle_id: self.vehicle_preference(vehicle_id)
            for vehicle_id in vehicle_ids
        }
        weights = np.full(horizon, 1.0 / horizon, dtype=np.float64)
        demand = np.zeros((len(node_list), self.service_count), dtype=np.float64)
        coverage_mass = np.zeros(len(node_list), dtype=np.float64)
        # 协同缓存只复用同一份严格因果预测；额外保存车辆-节点-未来时隙覆盖，
        # 让边际收益能够判断“两个节点是否服务同一批预测车辆”，而不是把
        # 每个节点的流行度简单相加。该元数据不改变预测值或历史登记顺序。
        predicted_coverage = np.zeros(
            (len(node_list), len(vehicle_ids), horizon), dtype=np.float64
        )
        predicted_vehicle_positions = np.zeros(
            (len(vehicle_ids), horizon, 2), dtype=np.float64
        )
        vehicle_prediction_available = np.zeros(len(vehicle_ids), dtype=bool)
        for node_index, node in enumerate(node_list):
            node_positions = self._node_positions(node, horizon)
            radius = self._node_radius(node, coverage_radius_m)
            for vehicle_index, vehicle_id in enumerate(vehicle_ids):
                positions = self._positions_from_prediction(
                    predictions.get(vehicle_id), horizon
                )
                if positions is None:
                    continue
                predicted_vehicle_positions[vehicle_index] = positions
                vehicle_prediction_available[vehicle_index] = True
                covered = np.linalg.norm(positions - node_positions, axis=1) <= radius + 1.0e-9
                predicted_coverage[node_index, vehicle_index] = covered.astype(float)
                if not np.any(covered):
                    continue
                cover_weights = weights * covered.astype(np.float64)
                coverage_mass[node_index] += float(cover_weights.sum())
                # 先把 H 个未来 slot 的覆盖权重求和，再乘车辆服务偏好向量。
                demand[node_index] += float(cover_weights.sum()) * preferences[vehicle_id]

        epsilon = np.finfo(np.float64).eps
        popularity = np.empty_like(demand)
        fallback_flags = np.zeros(len(node_list), dtype=bool)
        global_preference = self.global_preference
        records: list[dict[str, Any]] = []
        for node_index, node_id in enumerate(node_ids):
            if coverage_mass[node_index] <= 0.0:
                # 未来没有可预测车辆覆盖时使用全局历史先验，避免NaN或随机值。
                popularity[node_index] = global_preference
                fallback_flags[node_index] = True
            else:
                row = demand[node_index] + epsilon
                popularity[node_index] = row / float(row.sum())
            for service_id in service_ids:
                records.append(
                    {
                        "node_id": node_id,
                        "service_id": service_id,
                        "predicted_demand": float(demand[node_index, service_id]),
                        "predicted_popularity": float(popularity[node_index, service_id]),
                        "prediction_origin_slot": slot,
                    }
                )
        return {
            "node_ids": node_ids,
            "service_ids": service_ids,
            "prediction_origin_slot": slot,
            "future_slots": tuple(slot + index + 1 for index in range(horizon)),
            "predicted_demand": demand,
            "predicted_popularity": popularity,
            "coverage_mass": coverage_mass,
            "predicted_coverage": predicted_coverage,
            "vehicle_ids": vehicle_ids,
            "vehicle_prediction_available": vehicle_prediction_available,
            "vehicle_positions_m": predicted_vehicle_positions,
            "node_positions_m": np.asarray(
                [self._node_positions(node, horizon) for node in node_list],
                dtype=np.float64,
            ),
            "horizon_weights": weights,
            "prior_fallback": fallback_flags,
            "node_predictions": records,
            "vehicle_preferences": preferences,
        }

    predict = predict_node_demand

    def predict_global_sequence(
        self,
        task_vehicle_ids: Sequence[str],
        task_arrivals: Sequence[Sequence[bool]] | np.ndarray,
        task_service_types: Sequence[Sequence[int]] | np.ndarray,
    ) -> np.ndarray:
        """按严格因果顺序生成全局序列，便于离线兼容和短测试。"""

        arrivals = np.asarray(task_arrivals, dtype=bool)
        service_ids = np.asarray(task_service_types, dtype=np.int64)
        if arrivals.ndim != 2 or service_ids.shape != arrivals.shape:
            raise ValueError("请求历史必须是(T,V)数组")
        if arrivals.shape[1] != len(tuple(task_vehicle_ids)):
            raise ValueError("task_vehicle_ids数量与请求历史车辆维度不一致")
        result = np.empty((arrivals.shape[0], self.service_count), dtype=np.float64)
        for slot in range(arrivals.shape[0]):
            result[slot] = self.predict_global(slot)
            self.update_requests(
                slot,
                task_vehicle_ids,
                arrivals[slot],
                service_ids[slot],
            )
        return result


def get_service_vehicle_states(
    api: Any,
    config: Mapping[str, Any] | None = None,
    manager: VehicleRoleManager | None = None,
) -> dict[str, dict[str, Any]]:
    """读取当前服务车辆状态；传入manager时同时执行当前slot角色更新。"""

    loaded_config = config or load_config()
    if manager is not None:
        assignment = resolve_current_vehicle_roles(api, manager)
        return get_vehicle_states(api, assignment["service_vehicle_ids"])
    service_ids, _ = resolve_vehicle_roles(loaded_config)
    active_ids = {str(item) for item in api.vehicle.getIDList()}
    return get_vehicle_states(api, tuple(item for item in service_ids if item in active_ids))


def resolve_current_vehicle_roles(
    api: Any,
    manager: VehicleRoleManager,
    *,
    slot: int | None = None,
) -> dict[str, Any]:
    """Read the current SUMO active set and update one role manager."""

    if not isinstance(manager, VehicleRoleManager):
        raise TypeError("manager必须是VehicleRoleManager实例。")
    return manager.assign(get_vehicle_states(api), slot=slot)


def load_trajectory_npz(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """读取已验证的轨迹数组格式，不复制旧工程 Dataset 类。

    这里只接受数组级数据契约：时间、车辆 ID、位置和速度。后续环境仍会根据
    ``vehicle_ids`` 做列对齐，因而不会依赖 NPZ 中车辆列的偶然排列。
    """

    trajectory_path = _resolve_path(path)
    if trajectory_path is None or not trajectory_path.is_file():
        raise FileNotFoundError(f"轨迹NPZ不存在：{trajectory_path}")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        required = {"times", "vehicle_ids", "positions", "speeds"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"轨迹NPZ缺少字段：{sorted(missing)}")
        result = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if result["positions"].ndim != 3 or result["positions"].shape[-1] != 2:
        raise ValueError("轨迹positions必须是(T,V,2)。")
    if result["speeds"].shape != result["positions"].shape[:2]:
        raise ValueError("轨迹speeds必须是(T,V)。")
    result["vehicle_ids"] = tuple(str(item) for item in result["vehicle_ids"])
    # 离线轨迹同样以完整vehicle_ids作为候选全集；角色由稳定优先级确定，
    # 具体slot的当前角色由VehicleRoleManager根据当前状态更新。
    trajectory_config = config or load_config()
    service_ids, task_ids = resolve_vehicle_roles(
        trajectory_config, all_vehicle_ids=result["vehicle_ids"]
    )
    result["service_vehicle_ids"] = service_ids
    result["task_vehicle_ids"] = task_ids
    return result


def _service_popularity_from_requests(
    arrivals: np.ndarray,
    service_ids: np.ndarray,
    service_count: int,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    if arrivals.ndim != 2 or service_ids.shape != arrivals.shape:
        raise ValueError("请求到达和服务ID必须具有(T,V)形状。")
    if service_count <= 0:
        raise ValueError("service_count必须大于0。")
    previous = (
        np.full(service_count, 1.0 / service_count, dtype=np.float64)
        if initial is None
        else np.asarray(initial, dtype=np.float64).copy()
    )
    if previous.shape != (service_count,) or not np.all(np.isfinite(previous)):
        raise ValueError("initial流行度形状或数值无效。")
    previous = np.maximum(previous, 0.0)
    previous /= previous.sum() if previous.sum() > 0 else 1.0
    popularity = np.empty((arrivals.shape[0], service_count), dtype=np.float64)
    for index in range(arrivals.shape[0]):
        active_services = service_ids[index][np.asarray(arrivals[index], dtype=bool)]
        active_services = active_services[
            (active_services >= 0) & (active_services < service_count)
        ]
        if active_services.size:
            counts = np.bincount(active_services.astype(np.int64), minlength=service_count).astype(float)
            previous = counts / counts.sum()
        popularity[index] = previous
    return popularity


def build_gru_inputs(
    observed_popularity: Sequence[Sequence[float]] | np.ndarray,
    history_length: int,
    initial_popularity: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """构造论文 GRU 输入，形状为 ``(T, L_rho, F)``。

    对时刻 ``t``，输入序列只取 ``t-1`` 及以前最近 ``history_length`` 个观测
    流行度。episode 开头尚不存在的历史在左侧使用论文定义的初始服务流行度
    填充，避免把当前时隙请求泄漏到配置决策；只有调用方确实没有提供该先验时，
    才回退到均匀工程先验 ``[1/F,...,1/F]``。该函数只为离线 GRU baseline
    构造输入，不更新任何模型参数，也不参与正式环境需求预测。
    """

    history = np.asarray(observed_popularity, dtype=np.float32)
    if history.ndim != 2 or history.shape[0] <= 0:
        raise ValueError("observed_popularity必须是非空(T,F)数组。")
    if history_length <= 0:
        raise ValueError("history_length必须是正整数。")
    if not np.all(np.isfinite(history)) or np.any(history < 0):
        raise ValueError("observed_popularity必须是有限非负数组。")
    row_sums = history.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("observed_popularity每行必须有正和。")
    history = history / row_sums
    if initial_popularity is None:
        prior = np.full(history.shape[1], 1.0 / history.shape[1], dtype=np.float32)
    else:
        prior = np.asarray(initial_popularity, dtype=np.float32)
        if prior.shape != (history.shape[1],):
            raise ValueError("initial_popularity必须是(F,)向量。")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0):
            raise ValueError("initial_popularity必须是有限非负向量。")
        prior_sum = float(prior.sum())
        if prior_sum <= 0.0:
            raise ValueError("initial_popularity总和必须为正。")
        prior = prior / prior_sum
    result = np.empty((history.shape[0], history_length, history.shape[1]), dtype=np.float32)
    for time_index in range(history.shape[0]):
        for history_index in range(history_length):
            source_index = time_index - history_length + history_index
            result[time_index, history_index] = (
                prior if source_index < 0 else history[source_index]
            )
    return result


def generate_workload(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """按论文请求和Zipf语义生成最小工作负载。

    该函数只生成数据，不执行任务、计算 reward 或进行任何优化。

    到达、流行度、服务类型和输入规模使用独立随机数发生器，保证改变服务数
    ``F`` 时不会悄悄复用其他数据流的随机状态。``NO_TASK_SERVICE`` 只用于未到达
    任务；到达任务的 service type 一定从 ``0`` 到 ``F-1`` 取值。
    """

    loaded_config = config or load_config()
    simulation = _section(loaded_config, "simulation")
    services = _section(loaded_config, "services")
    workload = _section(loaded_config, "workload")
    slot_count = int(simulation["slot_count"])
    _, task_ids = resolve_vehicle_roles(loaded_config)
    service_count = int(services["count"])
    arrival_rng = np.random.default_rng(int(workload["arrival_seed"]))
    popularity_rng = np.random.default_rng(int(workload["popularity_seed"]))
    request_rng = np.random.default_rng(int(workload["service_request_seed"]))
    load_rng = np.random.default_rng(int(workload["task_input_seed"]))

    change_interval = int(workload["popularity_change_interval_slots"])
    if change_interval <= 0:
        raise ValueError("workload.popularity_change_interval_slots必须大于0。")
    zipf_exponent = _as_nonnegative(workload["zipf_exponent"], "workload.zipf_exponent")
    rankings: list[np.ndarray] = []
    popularity = np.empty((slot_count, service_count), dtype=np.float64)
    # 每个流行度块先生成一个随机排名，再按 Zipf 指数转换为概率向量。
    # 排名和概率被保存下来，便于后续复现实验和核对服务请求分布。
    for block_start in range(0, slot_count, change_interval):
        ranking = popularity_rng.permutation(service_count)
        rankings.append(ranking.copy())
        ranks = np.empty(service_count, dtype=np.float64)
        ranks[ranking] = np.arange(1, service_count + 1, dtype=np.float64)
        weights = np.power(ranks, -zipf_exponent)
        distribution = weights / weights.sum()
        block_end = min(slot_count, block_start + change_interval)
        popularity[block_start:block_end] = distribution

    arrival_probability = _as_float(
        workload["task_arrival_probability"], "workload.task_arrival_probability"
    )
    if not 0.0 <= arrival_probability <= 1.0:
        raise ValueError("workload.task_arrival_probability必须位于[0,1]。")
    arrivals = arrival_rng.random((slot_count, len(task_ids))) < arrival_probability
    service_ids = np.full((slot_count, len(task_ids)), NO_TASK_SERVICE, dtype=np.int64)
    input_bits = np.zeros_like(service_ids, dtype=np.float64)
    cpu_cycles = np.zeros_like(input_bits)
    input_min = _as_nonnegative(workload["task_input_size_min"], "workload.task_input_size_min")
    input_max = _as_positive(workload["task_input_size_max"], "workload.task_input_size_max")
    cycle_min = _as_nonnegative(workload["task_cpu_cycles_min"], "workload.task_cpu_cycles_min")
    cycle_max = _as_positive(workload["task_cpu_cycles_max"], "workload.task_cpu_cycles_max")
    if input_max < input_min or cycle_max < cycle_min:
        raise ValueError("任务输入或CPU范围无效。")
    # 请求服务类型只为到达车辆生成；未到达行保持 -1/0，避免把无任务误认为
    # 对某个服务的请求，也避免无效任务进入后续 matching。
    for time_index in range(slot_count):
        active = np.flatnonzero(arrivals[time_index])
        if active.size:
            service_ids[time_index, active] = request_rng.choice(
                service_count, size=active.size, p=popularity[time_index]
            )
            input_bits[time_index, active] = (
                load_rng.uniform(input_min, input_max, size=active.size) * 1_000_000.0
            )
            cpu_cycles[time_index, active] = (
                load_rng.uniform(cycle_min, cycle_max, size=active.size) * 1_000_000_000.0
            )

    observed_popularity = _service_popularity_from_requests(
        arrivals, service_ids, service_count
    )
    history_length = int(_section(loaded_config, "prediction")["history_length_slots"])
    gru_inputs = build_gru_inputs(
        observed_popularity,
        history_length,
        initial_popularity=popularity[0],
    )
    times = np.arange(
        float(simulation["optimization_begin_s"]),
        float(simulation["optimization_end_s"]),
        float(simulation["decision_interval_s"]),
        dtype=np.float64,
    )
    if times.shape != (slot_count,):
        raise ValueError("simulation时间范围与slot_count不一致。")
    return {
        "times": times,
        "task_vehicle_ids": task_ids,
        "task_arrivals": arrivals,
        "task_service_types": service_ids,
        "task_input_size_bits": input_bits,
        "task_cpu_cycles": cpu_cycles,
        "service_popularity": popularity,
        "observed_popularity": observed_popularity,
        "gru_inputs": gru_inputs,
        "service_rankings": np.asarray(rankings, dtype=np.int64),
    }


def load_workload_npz(
    path: str | os.PathLike[str],
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """读取旧工程工作负载数组，并为离线 GRU baseline 重构历史输入。"""

    workload_path = _resolve_path(path)
    if workload_path is None or not workload_path.is_file():
        raise FileNotFoundError(f"工作负载NPZ不存在：{workload_path}")
    with np.load(workload_path, allow_pickle=False) as archive:
        required = {"times", "task_arrivals", "task_service_types"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"工作负载NPZ缺少字段：{sorted(missing)}")
        result = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrivals = np.asarray(result["task_arrivals"], dtype=bool)
    service_ids = np.asarray(result["task_service_types"], dtype=np.int64)
    if arrivals.ndim != 2 or service_ids.shape != arrivals.shape:
        raise ValueError("工作负载task_arrivals和task_service_types必须是相同(T,V)形状。")
    loaded_config = config or load_config()
    service_count = int(loaded_config["services"]["count"])
    observed = _service_popularity_from_requests(arrivals, service_ids, service_count)
    history_length = int(loaded_config["prediction"]["history_length_slots"])
    result["task_arrivals"] = arrivals
    result["task_service_types"] = service_ids
    result["observed_popularity"] = observed
    initial_popularity = None
    if "service_popularity" in result:
        configured_popularity = np.asarray(result["service_popularity"], dtype=float)
        if configured_popularity.shape != (arrivals.shape[0], service_count):
            raise ValueError("service_popularity必须是(T,F)数组。")
        if not np.all(np.isfinite(configured_popularity)) or np.any(
            configured_popularity < 0.0
        ):
            raise ValueError("service_popularity必须是有限非负数组。")
        initial_popularity = configured_popularity[0]
    result["gru_inputs"] = build_gru_inputs(
        observed,
        history_length,
        initial_popularity=initial_popularity,
    )
    if "service_popularity" not in result:
        result["service_popularity"] = observed.copy()
    return result


class KalmanPredictor:
    """二维常速度 Kalman 预测器，对应论文的 ``[x,y,vx,vy]`` 状态。

    该预测器只服务于车辆位置短期预测：

    - ``predict`` 使用常速度状态转移得到下一时隙位置；
    - ``correct`` 使用 SUMO 实际位置和速度修正状态和协方差；
    - 旧 GRU 仅作为隔离 baseline；正式服务流行度由移动感知预测器负责。

    因此它不能替代论文中的服务流行度模型，也不参与 PPO 反向传播。
    """

    def __init__(
        self,
        vehicle_ids: Sequence[str],
        *,
        dt_s: float,
        measurement_noise_std_m: float = 1.0,
        acceleration_noise_std_mps2: float = 1.0,
        initial_position_variance_m2: float = 4.0,
        initial_velocity_variance_m2ps2: float = 4.0,
    ) -> None:
        self.vehicle_ids = tuple(str(item) for item in vehicle_ids)
        if not self.vehicle_ids or len(set(self.vehicle_ids)) != len(self.vehicle_ids):
            raise ValueError("vehicle_ids必须非空且不重复。")
        dt = _as_positive(dt_s, "dt_s")
        measurement_variance = _as_positive(
            measurement_noise_std_m, "measurement_noise_std_m"
        ) ** 2
        acceleration_variance = _as_positive(
            acceleration_noise_std_mps2, "acceleration_noise_std_mps2"
        ) ** 2
        self._transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # SUMO 同时提供位置和速度，因此测量向量与状态均为 [x,y,vx,vy]。
        # 11.pdf 未另给速度测量噪声；这里沿用现有测量噪声数值，不新增或调节参数。
        self._observation = np.eye(4, dtype=np.float64)
        control = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt], [dt, 0.0], [0.0, dt]])
        self._process_noise = acceleration_variance * np.einsum(
            "ij,kj->ik", control, control
        )
        self._measurement_noise = np.eye(4, dtype=np.float64) * measurement_variance
        self._states = np.zeros((len(self.vehicle_ids), 4), dtype=np.float64)
        self._covariances = np.zeros((len(self.vehicle_ids), 4, 4), dtype=np.float64)
        self._initialized = np.zeros(len(self.vehicle_ids), dtype=bool)
        self._index = {vehicle_id: index for index, vehicle_id in enumerate(self.vehicle_ids)}
        self._initial_position_variance = _as_positive(
            initial_position_variance_m2, "initial_position_variance_m2"
        )
        self._initial_velocity_variance = _as_positive(
            initial_velocity_variance_m2ps2, "initial_velocity_variance_m2ps2"
        )

    def initialize(
        self,
        positions_m: Sequence[Sequence[float]] | np.ndarray,
        velocities_mps: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> None:
        """用第一帧位置/速度建立每辆车的状态和初始协方差。"""

        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.shape != (len(self.vehicle_ids), 2):
            raise ValueError("Kalman初始位置必须是(V,2)。")
        velocities = (
            np.zeros_like(positions)
            if velocities_mps is None
            else np.asarray(velocities_mps, dtype=np.float64)
        )
        if velocities.shape != positions.shape:
            raise ValueError("Kalman初始速度必须是(V,2)。")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Kalman初始位置和速度必须有限。")
        self._states[:, :2] = positions
        self._states[:, 2:] = velocities
        self._covariances[:] = 0.0
        self._covariances[:, 0, 0] = self._initial_position_variance
        self._covariances[:, 1, 1] = self._initial_position_variance
        self._covariances[:, 2, 2] = self._initial_velocity_variance
        self._covariances[:, 3, 3] = self._initial_velocity_variance
        self._initialized[:] = True

    def predict(self) -> np.ndarray:
        """执行一次先验预测并返回所有车辆的二维预测位置。"""

        if not np.all(self._initialized):
            raise RuntimeError("KalmanPredictor必须先initialize。")
        predicted = np.empty((len(self.vehicle_ids), 2), dtype=np.float64)
        for index in range(len(self.vehicle_ids)):
            self._states[index] = np.einsum(
                "ij,j->i", self._transition, self._states[index]
            )
            propagated = np.einsum(
                "ij,jk->ik", self._transition, self._covariances[index]
            )
            self._covariances[index] = (
                np.einsum("ij,kj->ik", propagated, self._transition)
                + self._process_noise
            )
            predicted[index] = self._states[index, :2]
        return predicted.copy()

    def correct(
        self,
        positions_m: Sequence[Sequence[float]] | np.ndarray,
        velocities_mps: Sequence[Sequence[float]] | np.ndarray,
    ) -> None:
        """用同一时刻的 SUMO 实际位置和速度执行四维测量更新。"""

        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.asarray(velocities_mps, dtype=np.float64)
        if positions.shape != (len(self.vehicle_ids), 2):
            raise ValueError("Kalman观测位置必须是(V,2)。")
        if velocities.shape != positions.shape:
            raise ValueError("Kalman观测速度必须是(V,2)。")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Kalman观测位置和速度必须有限。")
        measurements = np.concatenate((positions, velocities), axis=1)
        identity = np.eye(4, dtype=np.float64)
        for index in range(len(self.vehicle_ids)):
            covariance = self._covariances[index]
            innovation = measurements[index] - np.einsum(
                "ij,j->i", self._observation, self._states[index]
            )
            projected_covariance = np.einsum(
                "ij,jk->ik", self._observation, covariance
            )
            innovation_covariance = (
                np.einsum("ij,kj->ik", projected_covariance, self._observation)
                + self._measurement_noise
            )
            gain_left = np.einsum("ij,kj->ik", covariance, self._observation)
            try:
                gain = np.linalg.solve(innovation_covariance.T, gain_left.T).T
            except np.linalg.LinAlgError as exc:
                raise ValueError("Kalman创新协方差矩阵不可逆。") from exc
            self._states[index] += np.einsum("ij,j->i", gain, innovation)
            residual = identity - np.einsum(
                "ij,jk->ik", gain, self._observation
            )
            residual_covariance = np.einsum(
                "ij,jk->ik", residual, covariance
            )
            gain_noise = np.einsum(
                "ij,jk->ik", gain, self._measurement_noise
            )
            self._covariances[index] = np.einsum(
                "ij,kj->ik", residual_covariance, residual
            ) + np.einsum("ij,kj->ik", gain_noise, gain)


def _velocity_vectors(
    speeds: np.ndarray,
    angles_deg: np.ndarray | None,
) -> np.ndarray:
    speed_values = np.asarray(speeds, dtype=np.float64)
    if angles_deg is None:
        return np.zeros((*speed_values.shape, 2), dtype=np.float64)
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    if angles.shape != speed_values.shape:
        raise ValueError("angles和speeds必须具有相同形状。")
    # SUMO 使用导航角：0°朝北、90°朝东；平面坐标为 +x 东、+y 北。
    return np.stack((speed_values * np.sin(angles), speed_values * np.cos(angles)), axis=-1)


def predict_vehicle_positions(
    positions_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    speeds_mps: Sequence[Sequence[float]] | np.ndarray,
    *,
    angles_deg: Sequence[Sequence[float]] | np.ndarray | None = None,
    dt_s: float,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """根据实际观测序列得到每个时隙的预测位置。

    返回数组与输入 ``positions_m`` 具有相同的 ``(T,V,2)`` 形状。第一个时隙
    是缺少前置观测的离线初始化边界；之后每个时隙先由上一时隙已校正状态预测，
    再在该时隙结束后用其位置和速度校正，避免后续预测读取未来观测。正式在线
    SUMO 路径会额外读取第一个配置时隙之前的观测，不使用该边界回退。
    """

    positions = np.asarray(positions_m, dtype=np.float64)
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions必须是(T,V,2)。")
    if speeds.shape != positions.shape[:2]:
        raise ValueError("speeds必须是(T,V)。")
    if positions.shape[0] == 0:
        return np.empty_like(positions)
    prediction_cfg = _section(config or load_config(), "prediction")
    kalman_cfg = _section(prediction_cfg, "kalman")
    vehicle_ids = tuple(str(index) for index in range(positions.shape[1]))
    predictor = KalmanPredictor(
        vehicle_ids,
        dt_s=dt_s,
        measurement_noise_std_m=kalman_cfg["measurement_noise_std_m"],
        acceleration_noise_std_mps2=kalman_cfg["acceleration_noise_std_mps2"],
        initial_position_variance_m2=kalman_cfg["initial_position_variance_m2"],
        initial_velocity_variance_m2ps2=kalman_cfg["initial_velocity_variance_m2ps2"],
    )
    angle_array = None if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
    velocity_vectors = _velocity_vectors(speeds, angle_array)
    predictor.initialize(positions[0], velocity_vectors[0])
    predicted = np.empty_like(positions)
    predicted[0] = positions[0]
    for time_index in range(1, positions.shape[0]):
        predicted[time_index] = predictor.predict()
        predictor.correct(positions[time_index], velocity_vectors[time_index])

    if bool(kalman_cfg.get("clip_to_region", False)):
        region = _section(_section(config or load_config(), "uav"), "region")
        predicted[..., 0] = np.clip(
            predicted[..., 0], float(region["x_min_m"]), float(region["x_max_m"])
        )
        predicted[..., 1] = np.clip(
            predicted[..., 1], float(region["y_min_m"]), float(region["y_max_m"])
        )
    return predicted


def build_gru_model(config: Mapping[str, Any] | None = None) -> Any:
    """构造论文需要的最小 ``GRU+Linear+Softmax`` 预测器，不执行训练。

    GRU 的 ``input_size`` 和输出层大小都从 ``services.count`` 读取，因此 F=10
    或后续敏感性实验不会复用固定的 F=20/180 维度。这里将参数设为冻结状态，
    真正的正式实验必须由外部提供已经训练好的、与当前 F 一致的 checkpoint。
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on selected environment
        raise RuntimeError("构造GRU需要PyTorch；本轮不自动安装或训练模型。") from exc
    loaded_config = config or load_config()
    service_count = int(_section(loaded_config, "services")["count"])
    gru_config = _section(_section(loaded_config, "prediction"), "gru")
    hidden_size = int(gru_config["hidden_size"])
    num_layers = int(gru_config["num_layers"])
    if hidden_size <= 0 or num_layers <= 0:
        raise ValueError("GRU hidden_size和num_layers必须为正整数。")

    class PopularityGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # batch_first=True 使输入直接采用 (batch, history_length, service_count)。
            # 输出只取最后一个时间步，因为它代表当前时隙对下一决策的流行度预测。
            self.gru = nn.GRU(
                input_size=service_count,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_size, service_count)

        def forward(self, inputs: Any) -> Any:
            sequence, _ = self.gru(inputs)
            # Softmax 把线性层输出变成总和为1的服务流行度分布。
            return torch.softmax(self.output(sequence[:, -1, :]), dim=-1)

    model = PopularityGRU()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def predict_service_popularity(model: Any, gru_inputs: np.ndarray) -> np.ndarray:
    """使用已经准备好的冻结 GRU 推理，不包含训练和 ``optimizer.step``。

    该函数只执行前向计算并返回 NumPy 数组，便于环境状态构造和策略网络读取。
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on selected environment
        raise RuntimeError("GRU推理需要PyTorch。") from exc
    inputs = np.asarray(gru_inputs, dtype=np.float32)
    if inputs.ndim != 3:
        raise ValueError("gru_inputs必须是(T,L,F)。")
    with torch.no_grad():
        output = model(torch.from_numpy(inputs)).detach().cpu().numpy()
    if output.shape != (inputs.shape[0], inputs.shape[2]):
        raise ValueError("GRU输出维度与服务数量不一致。")
    return output.astype(np.float64, copy=False)


def load_gru_checkpoint(
    checkpoint_file: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
) -> Any:
    """加载外部预训练 GRU 并冻结参数；不创建或训练 checkpoint。

    checkpoint 可以直接是 state dict，也可以是包含 ``state_dict`` 字段的映射。
    加载完成后模型被切换到 ``eval`` 模式，防止 dropout 或训练态统计影响论文
    中作为固定预测器的语义。
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on selected environment
        raise RuntimeError("加载GRU checkpoint需要PyTorch。") from exc
    checkpoint_path = _resolve_path(checkpoint_file)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"GRU checkpoint不存在：{checkpoint_path}")
    model = build_gru_model(config)
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state_dict)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def load_data(config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取路网并准备后续 environment 需要的最小 slot 数据。

    数据来源优先级是：显式配置的轨迹/工作负载 NPZ，否则使用当前项目内 SUMO
    路网和按配置种子生成的工作负载。服务流行度预测使用严格因果的移动感知
    预测器；历史 GRU 代码仅保留为隔离的离线基线，不再被正式环境/训练调用。
    """

    config = load_config(config_path)
    scene = load_sumo_scene(config)
    data_cfg = _section(config, "data")
    trajectory = (
        load_trajectory_npz(data_cfg["trajectory_file"], config)
        if data_cfg.get("trajectory_file")
        else None
    )
    workload = (
        load_workload_npz(data_cfg["workload_file"], config)
        if data_cfg.get("workload_file")
        else generate_workload(config)
    )
    # 如果提供了离线轨迹，提前计算 Kalman 预测；如果当前直接读取 SUMO，则由
    # MECEnvironment 在 reset/step 中根据实时观测维护预测器状态。
    predicted_positions = None
    if trajectory is not None:
        predicted_positions = predict_vehicle_positions(
            trajectory["positions"],
            trajectory["speeds"],
            angles_deg=trajectory.get("angles"),
            dt_s=float(_section(config, "simulation")["decision_interval_s"]),
            config=config,
        )

    # 正式环境在每个 slot 在线维护预测器；这里不预扫描全量请求生成序列，
    # 避免离线 data source 看见未来请求。predicted_popularity 由环境动态提供。
    predicted_popularity = None
    prediction_source = "mobility_aware_popularity"

    return {
        "config": config,
        "sumo_scene": scene,
        "trajectory": trajectory,
        "workload": workload,
        "predicted_positions": predicted_positions,
        "predicted_popularity": predicted_popularity,
        "prediction_source": prediction_source,
    }


def build_causal_popularity_sequence(
    workload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    """为离线 data_source 生成严格因果的全局服务流行度序列。

    该序列用于保持旧数据接口的 F 维兼容；在线环境会继续使用同一个预测器
    结合道路约束覆盖生成节点级需求/流行度。每个 slot 的请求都在预测完成后
    才写入历史。
    """

    services = _section(config, "services")
    trajectory_config = config.get("trajectory_prediction", {})
    if not isinstance(trajectory_config, Mapping):
        trajectory_config = {}
    popularity_config = config.get("popularity_prediction", {})
    if not isinstance(popularity_config, Mapping):
        popularity_config = {}
    predictor = MobilityAwarePopularityPredictor(
        int(services["count"]),
        history_decay=float(popularity_config.get("history_decay", 0.8)),
        preference_smoothing=float(
            popularity_config.get("preference_smoothing", 1.0)
        ),
        horizon_slots=int(trajectory_config.get("horizon_slots", 3)),
    )
    return predictor.predict_global_sequence(
        workload["task_vehicle_ids"],
        np.asarray(workload["task_arrivals"], dtype=bool),
        np.asarray(workload["task_service_types"], dtype=np.int64),
    )


def get_slot_data(data: Mapping[str, Any], slot_index: int) -> dict[str, Any]:
    """以简单 dict 返回 environment 单个 slot 需要的数据。

    本函数只做时间索引和字段名兼容，不修改底层数组。返回的数组全部复制，
    这样 solver 或诊断代码修改一个时隙的临时数据时，不会污染完整 workload。
    """

    workload = data["workload"]
    times = np.asarray(workload["times"])
    index = int(slot_index)
    if index < 0 or index >= len(times):
        raise IndexError(f"slot_index超出范围：{index}")
    input_key = (
        "task_input_size_bits"
        if "task_input_size_bits" in workload
        else "task_input_sizes_bits"
    )
    cycle_key = (
        "task_cpu_cycles"
        if "task_cpu_cycles" in workload
        else "task_cpu_cycles_per_task"
    )
    trajectory = data.get("trajectory")
    actual_positions = None if trajectory is None else np.asarray(trajectory["positions"])[index].copy()
    predicted_positions = data.get("predicted_positions")
    predicted = None if predicted_positions is None else np.asarray(predicted_positions)[index].copy()
    resolved = data["config"].get("_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("data.config必须包含场景初始化时生成的_resolved车辆集合。")
    fixed_service_ids = _vehicle_ids(
        resolved.get("service_vehicle_ids"),
        "_resolved.service_vehicle_ids",
        allow_empty=True,
    )
    return {
        "time_s": float(times[index]),
        "actual_positions_m": actual_positions,
        "predicted_positions_m": predicted,
        # slot切片只读取场景初始化时固定的集合，不重新扫描TraCI当前车辆。
        "service_vehicle_ids": fixed_service_ids,
        "task_vehicle_ids": tuple(workload["task_vehicle_ids"]),
        "task_arrivals": np.asarray(workload["task_arrivals"])[index].copy(),
        "task_service_types": np.asarray(workload["task_service_types"])[index].copy(),
        "task_input_size_bits": np.asarray(workload[input_key])[index].copy(),
        "task_cpu_cycles": np.asarray(workload[cycle_key])[index].copy(),
        "observed_popularity": np.asarray(workload["observed_popularity"])[index].copy(),
        "predicted_popularity": (
            None
            if data.get("predicted_popularity") is None
            else np.asarray(data["predicted_popularity"])[index].copy()
        ),
        "gru_input": np.asarray(workload["gru_inputs"])[index].copy(),
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "KalmanPredictor",
    "MobilityAwarePopularityPredictor",
    "NO_TASK_SERVICE",
    "RoadConstrainedVehiclePredictor",
    "RoadTransitionModel",
    "VehicleRoleManager",
    "build_gru_inputs",
    "build_gru_model",
    "build_causal_popularity_sequence",
    "close",
    "generate_workload",
    "get_service_vehicle_states",
    "get_slot_data",
    "get_vehicle_states",
    "load_config",
    "load_data",
    "load_gru_checkpoint",
    "load_sumo_scene",
    "load_trajectory_npz",
    "load_workload_npz",
    "predict_service_popularity",
    "predict_vehicle_positions",
    "resolve_current_vehicle_roles",
    "resolve_vehicle_roles",
    "reset",
    "stable_priority",
    "start_sumo",
    "step",
]
