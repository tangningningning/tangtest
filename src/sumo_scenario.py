#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 SUMO 交通、采集轨迹并固定划分 TV/SV 角色。

本文件只调用 SUMO 官方工具和 Python API，没有复制 SUMO 的实现代码：

* randomTrips.py：
  https://github.com/eclipse-sumo/sumo/blob/main/tools/randomTrips.py
* TraCI Python 客户端：
  https://github.com/eclipse-sumo/sumo/tree/main/tools/traci
* TraCI 车辆状态接口文档：
  https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html

SUMO 项目采用 Eclipse Public License 2.0。本项目通过 API 调用 SUMO，未移植
其源码；因此这里不需要复制 SUMO 源文件，只在注释中保留来源和许可证说明。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# 主场景配置区
# ============================================================================


@dataclass(frozen=True)
class ScenarioConfig:
    """本阶段唯一的场景参数集合。

    episode_time_s 表示 warmup 结束后的正式 episode 时长，正式时隙为：
        [warmup_time_s, warmup_time_s + episode_time_s)

    randomTrips 参数也集中在这里，便于后续更换需求密度而不改动采集逻辑。
    """

    dt_s: float = 1.0
    warmup_time_s: float = 100.0
    episode_time_s: float = 300.0
    num_research_vehicles: int = 12
    num_service_vehicles: int = 3
    num_task_vehicles: int = 9
    random_seed: int = 4401

    # 仅在 data/ 下没有可直接运行的交通需求时使用官方 randomTrips.py。
    random_trips_period_s: float = 1.0
    random_trips_departure_window_s: float = 60.0
    random_trips_min_distance_m: float = 4000.0
    random_trips_intermediate: int = 2
    random_trips_maxtries: int = 1000

    @property
    def simulation_end_s(self) -> float:
        return self.warmup_time_s + self.episode_time_s

    def validate(self) -> None:
        """校验时间网格和角色数量，避免产生含义不清的正式时隙。"""
        if self.dt_s <= 0:
            raise ValueError("dt 必须大于 0 秒。")
        if self.warmup_time_s < 0:
            raise ValueError("warmup_time 必须不小于 0 秒。")
        if self.episode_time_s <= 0:
            raise ValueError("episode_time 必须大于 0 秒。")

        for name, value in (
            ("warmup_time", self.warmup_time_s),
            ("episode_time", self.episode_time_s),
        ):
            steps = value / self.dt_s
            if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{name}={value} 不是 dt={self.dt_s} 的整数倍，"
                    "无法定义完整的正式时隙。"
                )

        if self.num_research_vehicles <= 0:
            raise ValueError("num_research_vehicles 必须大于 0。")
        if self.num_service_vehicles < 0:
            raise ValueError("num_service_vehicles 不能为负数。")
        if self.num_task_vehicles < 0:
            raise ValueError("num_task_vehicles 不能为负数。")
        if (
            self.num_service_vehicles + self.num_task_vehicles
            != self.num_research_vehicles
        ):
            raise ValueError(
                "角色数量必须满足 num_service_vehicles + "
                "num_task_vehicles = num_research_vehicles。"
            )

        if self.random_trips_period_s <= 0:
            raise ValueError("randomTrips period 必须大于 0。")
        if self.random_trips_departure_window_s <= 0:
            raise ValueError("randomTrips departure window 必须大于 0。")
        if self.random_trips_min_distance_m < 0:
            raise ValueError("randomTrips min-distance 不能为负数。")
        if self.random_trips_intermediate < 0:
            raise ValueError("randomTrips intermediate 不能为负数。")
        if self.random_trips_maxtries <= 0:
            raise ValueError("randomTrips maxtries 必须大于 0。")


DEFAULT_CONFIG = ScenarioConfig()

TRAJECTORY_FIELDS = (
    "time_s",
    "vehicle_id",
    "x_m",
    "y_m",
    "speed_mps",
    "acceleration_mps2",
    "road_id",
    "lane_id",
    "lane_position_m",
    "route_id",
    "route_edges",
    "distance_m",
)


@dataclass(frozen=True)
class TrafficSource:
    """一次 SUMO 运行所使用的路网、需求和可选配置。"""

    network_file: Path
    route_files: tuple[Path, ...]
    configuration_file: Path | None = None
    generated_by_random_trips: bool = False


# ============================================================================
# 路径、SUMO 环境和文件发现
# ============================================================================


def find_project_root() -> Path:
    """从脚本位置或当前工作目录找到包含 data/ 的项目根目录。"""
    candidates = (
        Path(__file__).resolve().parents[1],
        Path.cwd().resolve(),
    )
    for candidate in candidates:
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "无法定位项目根目录：期望在项目根目录下找到 data/。"
    )


def _resolve_from_project(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _is_readable_file(path: Path) -> bool:
    """用实际打开操作判断文件是否可被当前进程读到。"""
    try:
        with path.open("rb") as stream:
            stream.read(1)
        return True
    except (OSError, PermissionError):
        return False


def discover_sumo_files(project_root: Path) -> dict[str, list[Path]]:
    """自动发现 data/ 下的 SUMO 路网、配置、路线和 trips 文件。"""
    data_root = project_root / "data"
    networks: list[Path] = []
    configurations: list[Path] = []
    route_files: list[Path] = []

    for path in data_root.rglob("*"):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue

        name = path.name.lower()
        if name.endswith(".net.xml") or name.endswith(".net.xml.gz"):
            networks.append(path.resolve(strict=False))
        elif name.endswith(".sumocfg"):
            configurations.append(path.resolve(strict=False))
        elif (
            name.endswith(".rou.xml")
            or name.endswith(".rou.xml.gz")
            or name.endswith(".trips.xml")
            or name.endswith(".trips.xml.gz")
        ):
            route_files.append(path.resolve(strict=False))

    key = lambda path: path.relative_to(project_root).as_posix().lower()
    return {
        "networks": sorted(networks, key=key),
        "configurations": sorted(configurations, key=key),
        "route_files": sorted(route_files, key=key),
    }


def _format_inventory_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def print_sumo_inventory(project_root: Path, inventory: dict[str, list[Path]]) -> None:
    """把自动发现结果打印出来，便于确认没有覆盖原始文件。"""
    print("发现的 SUMO 文件：")
    for label, key in (
        ("路网", "networks"),
        ("配置", "configurations"),
        ("路线/需求", "route_files"),
    ):
        paths = inventory[key]
        print(f"  {label} ({len(paths)}):")
        for path in paths:
            readable = "可读" if _is_readable_file(path) else "不可读"
            print(f"    - {_format_inventory_path(project_root, path)} [{readable}]")


def find_sumo_home(explicit: str | None = None) -> Path:
    """按显式参数、SUMO_HOME、PATH 的顺序找到 SUMO 安装根目录。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    environment_home = os.environ.get("SUMO_HOME")
    if environment_home:
        candidates.append(Path(environment_home).expanduser())

    sumo_executable = shutil.which("sumo")
    if sumo_executable:
        executable_path = Path(sumo_executable).resolve(strict=False)
        if executable_path.parent.name.lower() == "bin":
            candidates.append(executable_path.parent.parent)
        else:
            candidates.append(executable_path.parent)

    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if (candidate / "tools" / "randomTrips.py").is_file():
            return candidate

    raise FileNotFoundError(
        "找不到 SUMO 安装。请设置 SUMO_HOME，或使用 --sumo-home 指定安装目录。"
    )


def load_sumo_modules(sumo_home: Path) -> tuple[Any, Any]:
    """加载 SUMO 官方 sumolib 和 traci Python 模块。"""
    tools_dir = str(sumo_home / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    try:
        import sumolib  # type: ignore[import-not-found]
        import traci  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            f"无法从 {tools_dir} 导入 SUMO 官方 sumolib/traci。"
        ) from exc
    return sumolib, traci


def find_sumo_binary(sumolib: Any, gui: bool) -> str:
    """优先使用 sumolib.checkBinary 找到 sumo 或 sumo-gui。"""
    binary_name = "sumo-gui" if gui else "sumo"
    try:
        return str(sumolib.checkBinary(binary_name))
    except Exception as exc:
        fallback = shutil.which(binary_name)
        if fallback:
            return fallback
        raise FileNotFoundError(f"找不到 SUMO 可执行文件：{binary_name}") from exc


# ============================================================================
# 现有 SUMO 配置/需求解析与官方 randomTrips 回退
# ============================================================================


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_value(root: ET.Element, tag_name: str) -> str | None:
    for element in root.iter():
        if _local_name(element.tag) == tag_name:
            value = element.get("value")
            if value:
                return value.strip()
    return None


def _split_file_references(value: str) -> list[str]:
    # SUMO 配置通常使用逗号；兼容少量手工配置中的分号分隔。
    return [token.strip() for token in re.split(r"[,;]", value) if token.strip()]


def _resolve_config_reference(config_file: Path, value: str) -> list[Path]:
    references: list[Path] = []
    for token in _split_file_references(value):
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = config_file.parent / path
        references.append(path.resolve(strict=False))
    return references


def _source_from_config(config_file: Path) -> TrafficSource | None:
    """读取已有 .sumocfg，仅在所有引用文件都实际可读时复用。"""
    try:
        root = ET.parse(config_file).getroot()
    except (OSError, ET.ParseError):
        return None

    network_value = _xml_value(root, "net-file")
    route_value = _xml_value(root, "route-files")
    if not network_value or not route_value:
        return None

    networks = _resolve_config_reference(config_file, network_value)
    routes = _resolve_config_reference(config_file, route_value)
    if len(networks) != 1 or not routes:
        return None
    if not _is_readable_file(networks[0]):
        return None
    if not all(_is_readable_file(path) for path in routes):
        return None

    return TrafficSource(
        network_file=networks[0],
        route_files=tuple(routes),
        configuration_file=config_file,
        generated_by_random_trips=False,
    )


def _choose_network(
    project_root: Path,
    inventory: dict[str, list[Path]],
    explicit_network: str | None,
) -> Path:
    if explicit_network:
        network = _resolve_from_project(project_root, explicit_network)
        if not _is_readable_file(network):
            raise FileNotFoundError(f"指定的 SUMO 路网不可读：{network}")
        return network

    for network in inventory["networks"]:
        if _is_readable_file(network):
            return network

    raise FileNotFoundError("data/ 下没有可读的 *.net.xml 或 *.net.xml.gz 路网文件。")


def _choose_existing_source(
    project_root: Path,
    inventory: dict[str, list[Path]],
    explicit_config: str | None,
    explicit_network: str | None,
    explicit_routes: list[str],
) -> TrafficSource | None:
    """优先复用可运行配置，其次复用可读的独立 route/trips 文件。"""
    if explicit_config:
        config_file = _resolve_from_project(project_root, explicit_config)
        source = _source_from_config(config_file)
        if source is None:
            raise RuntimeError(
                f"指定的 SUMO 配置不可直接运行或引用文件不可读：{config_file}"
            )
        return source

    # 有显式 net/route 时按用户指定组合，不使用别的配置隐式覆盖它们。
    if explicit_network or explicit_routes:
        network = _choose_network(project_root, inventory, explicit_network)
        routes = tuple(
            _resolve_from_project(project_root, value) for value in explicit_routes
        )
        if not routes:
            return None
        if not all(_is_readable_file(path) for path in routes):
            raise FileNotFoundError("至少一个指定的 SUMO route/trips 文件不可读。")
        return TrafficSource(network_file=network, route_files=routes)

    # 先尝试已有配置；配置中的相对路径以 .sumocfg 所在目录为基准。
    for config_file in inventory["configurations"]:
        source = _source_from_config(config_file)
        if source is not None:
            return source

    network = _choose_network(project_root, inventory, None)
    generated_dir = (project_root / "data" / "generated").resolve(strict=False)

    # 已有交通需求优先于本模块之前生成的文件；排序保证跨平台稳定。
    candidates = [
        path
        for path in inventory["route_files"]
        if path.parent != generated_dir
    ] + [
        path
        for path in inventory["route_files"]
        if path.parent == generated_dir
    ]
    for route_file in candidates:
        if _is_readable_file(route_file):
            return TrafficSource(network_file=network, route_files=(route_file,))

    return None


def generate_random_trips(
    project_root: Path,
    sumo_home: Path,
    network_file: Path,
    output_dir: Path,
    config: ScenarioConfig,
) -> TrafficSource:
    """调用 SUMO 官方 randomTrips.py 生成独立的 .rou.xml 需求文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    route_file = (output_dir / "sumo_demand.rou.xml").resolve(strict=False)

    if route_file.exists():
        if _is_readable_file(route_file):
            print(f"复用本模块之前生成的交通需求：{route_file}")
            return TrafficSource(
                network_file=network_file,
                route_files=(route_file,),
                generated_by_random_trips=True,
            )
        raise PermissionError(f"已有生成需求文件不可读，不覆盖它：{route_file}")

    random_trips = sumo_home / "tools" / "randomTrips.py"
    if not random_trips.is_file():
        raise FileNotFoundError(f"SUMO 官方 randomTrips.py 不存在：{random_trips}")

    # 让至少一批车辆在 warm-up 前后进入路网；后续所有车辆仍由 SUMO 正常运行。
    departure_end = max(
        config.warmup_time_s,
        config.random_trips_departure_window_s,
        config.dt_s,
    )

    command = [
        sys.executable,
        str(random_trips),
        "-n",
        str(network_file),
        "-o",
        "{trip_file}",
        "-r",
        str(route_file),
        "-b",
        "0",
        "-e",
        f"{departure_end:.9g}",
        "-p",
        f"{config.random_trips_period_s:.9g}",
        "-s",
        str(config.random_seed),
        "--min-distance",
        f"{config.random_trips_min_distance_m:.9g}",
        "-i",
        str(config.random_trips_intermediate),
        "--maxtries",
        str(config.random_trips_maxtries),
        "--vehicle-class",
        "passenger",
    ]

    environment = os.environ.copy()
    environment["SUMO_HOME"] = str(sumo_home)

    print("没有可直接运行的可读交通需求，调用 SUMO 官方 randomTrips.py ...")
    with tempfile.TemporaryDirectory(prefix="sumo_randomTrips_") as temporary:
        trip_file = Path(temporary) / "sumo_demand.trips.xml"
        command = [str(trip_file) if item == "{trip_file}" else item for item in command]
        subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=True,
        )

    if not _is_readable_file(route_file):
        raise RuntimeError(
            f"randomTrips.py 执行结束但没有生成可读路线文件：{route_file}"
        )

    return TrafficSource(
        network_file=network_file,
        route_files=(route_file,),
        generated_by_random_trips=True,
    )


# ============================================================================
# SUMO 启动和 TraCI 轨迹采集
# ============================================================================


def _number(value: float) -> str:
    return f"{value:.9g}"


def build_sumo_command(
    binary: str,
    source: TrafficSource,
    config: ScenarioConfig,
) -> list[str]:
    """构造无 GUI 默认的 SUMO 命令，不包含任何车辆控制命令。"""
    if source.configuration_file is not None:
        command = [binary, "-c", str(source.configuration_file)]
    else:
        command = [
            binary,
            "--net-file",
            str(source.network_file),
            "--route-files",
            ",".join(str(path) for path in source.route_files),
        ]

    # 命令行参数覆盖已有配置中的时间设置，但不修改配置文件本身。
    command.extend(
        [
            "--begin",
            "0",
            "--step-length",
            _number(config.dt_s),
            "--end",
            _number(config.simulation_end_s),
            "--seed",
            str(config.random_seed),
            "--no-step-log",
            "true",
        ]
    )
    return command


def _capture_vehicle_row(
    traci: Any,
    vehicle_id: str,
    time_s: float,
    route_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """通过 TraCI 官方 getter 读取一个车辆在当前 simulation step 的状态。"""
    if vehicle_id not in route_cache:
        route_id = str(traci.vehicle.getRouteID(vehicle_id))
        route_edges = tuple(str(edge) for edge in traci.vehicle.getRoute(vehicle_id))
        route_cache[vehicle_id] = {
            "route_id": route_id,
            "route_edges": route_edges,
        }

    planned_route = route_cache[vehicle_id]
    x_m, y_m = traci.vehicle.getPosition(vehicle_id)

    return {
        "time_s": float(time_s),
        "vehicle_id": vehicle_id,
        "x_m": float(x_m),
        "y_m": float(y_m),
        "speed_mps": float(traci.vehicle.getSpeed(vehicle_id)),
        "acceleration_mps2": float(traci.vehicle.getAcceleration(vehicle_id)),
        "road_id": str(traci.vehicle.getRoadID(vehicle_id)),
        "lane_id": str(traci.vehicle.getLaneID(vehicle_id)),
        "lane_position_m": float(traci.vehicle.getLanePosition(vehicle_id)),
        "route_id": planned_route["route_id"],
        "route_edges": " ".join(planned_route["route_edges"]),
        # TraCI getDistance 是 SUMO 已提供的累计行驶距离，单位为米。
        "distance_m": float(traci.vehicle.getDistance(vehicle_id)),
    }


def collect_trajectories(
    traci: Any,
    command: list[str],
    config: ScenarioConfig,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[str], list[float]]:
    """启动 SUMO，逐 step 采集所有车辆并在 finally 中关闭连接。"""
    rows: list[dict[str, Any]] = []
    route_cache: dict[str, dict[str, Any]] = {}
    seen_vehicle_ids: set[str] = set()
    observed_times: list[float] = []
    started = False

    print("启动 SUMO：")
    print("  " + " ".join(command))
    try:
        traci.start(command)
        started = True
        previous_time = 0.0

        while previous_time < config.simulation_end_s - 1e-8:
            # 只推进仿真，不调用 setRoute/changeTarget/rerouteTraveltime 等控制 API。
            traci.simulationStep()
            time_s = float(traci.simulation.getTime())
            if time_s > config.simulation_end_s + 1e-7:
                break

            if observed_times and not math.isclose(
                time_s - observed_times[-1],
                config.dt_s,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "SUMO 实际 step-length 与配置 dt 不一致："
                    f"前一时刻={observed_times[-1]}，当前={time_s}，"
                    f"dt={config.dt_s}。"
                )

            observed_times.append(time_s)
            for vehicle_id in traci.vehicle.getIDList():
                row = _capture_vehicle_row(traci, str(vehicle_id), time_s, route_cache)
                rows.append(row)
                seen_vehicle_ids.add(str(vehicle_id))

            previous_time = time_s
            if time_s >= config.simulation_end_s - 1e-7:
                break
            if traci.simulation.getMinExpectedNumber() <= 0:
                # 没有仍在路上或等待出发的车辆时，继续推进不会产生轨迹。
                break
    finally:
        if started:
            try:
                traci.close()
            except Exception:
                pass

    return rows, route_cache, seen_vehicle_ids, observed_times


# ============================================================================
# 正式区间完整轨迹筛选和路线分层角色选择
# ============================================================================


def _formal_step_keys(config: ScenarioConfig) -> set[int]:
    """返回正式时隙的整数 step 索引。

    SUMO 的第一次 simulationStep 通常出现在 dt 时刻。warmup>0 时使用
    [warmup, end)；warmup=0 时使用 [dt, end]，仍保持 episode/dt 个样本。
    """
    warmup_steps = int(round(config.warmup_time_s / config.dt_s))
    end_steps = int(round(config.simulation_end_s / config.dt_s))
    start_step = max(1, warmup_steps)
    end_step_exclusive = end_steps if warmup_steps > 0 else end_steps + 1
    return set(range(start_step, end_step_exclusive))


def _step_key(time_s: float, dt_s: float) -> int:
    return int(round(time_s / dt_s))


def _valid_trajectory_row(row: dict[str, Any]) -> bool:
    numeric_fields = (
        "time_s",
        "x_m",
        "y_m",
        "speed_mps",
        "acceleration_mps2",
        "lane_position_m",
        "distance_m",
    )
    for field in numeric_fields:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False

    if row["speed_mps"] < -1e-6 or row["distance_m"] < -1e-6:
        return False
    for field in ("vehicle_id", "road_id", "lane_id", "route_edges"):
        if not str(row.get(field, "")).strip():
            return False
    return True


def find_complete_vehicles(
    rows: Iterable[dict[str, Any]],
    route_cache: dict[str, dict[str, Any]],
    config: ScenarioConfig,
) -> tuple[list[str], dict[str, dict[str, Any]], set[int]]:
    """只保留正式区间每个时隙都有有效记录的车辆。"""
    formal_steps = _formal_step_keys(config)
    rows_by_vehicle: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        step = _step_key(float(row["time_s"]), config.dt_s)
        if step not in formal_steps:
            continue
        vehicle_id = str(row["vehicle_id"])
        # 同一车辆同一正式时隙出现重复记录时，视为不完整而不是静默覆盖。
        if step in rows_by_vehicle[vehicle_id]:
            rows_by_vehicle[vehicle_id][step] = {
                "__duplicate__": True,
            }
        else:
            rows_by_vehicle[vehicle_id][step] = row

    complete_ids: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    for vehicle_id in sorted(rows_by_vehicle):
        time_rows = rows_by_vehicle[vehicle_id]
        if set(time_rows) != formal_steps:
            continue
        if any(
            row.get("__duplicate__") is True
            or not _valid_trajectory_row(row)
            for row in time_rows.values()
        ):
            continue

        complete_ids.append(vehicle_id)
        route = route_cache.get(vehicle_id, {})
        route_id = str(route.get("route_id", ""))
        route_edges = tuple(str(edge) for edge in route.get("route_edges", ()))
        metadata[vehicle_id] = {
            "route_id": route_id,
            "route_edges": route_edges,
            "route_group": route_group_key(route_id, route_edges),
            "formal_record_count": len(time_rows),
        }

    return complete_ids, metadata, formal_steps


def route_group_key(route_id: str, route_edges: tuple[str, ...]) -> str:
    """优先按 route_id 分组；缺少 route_id 时按完整 edge sequence 分组。"""
    if route_id.strip():
        return f"route_id:{route_id}"
    if route_edges:
        return "edges:" + " ".join(route_edges)
    return "unknown"


def stratified_random_choice(
    vehicle_ids: Iterable[str],
    route_metadata: dict[str, dict[str, Any]],
    count: int,
    seed: int,
) -> list[str]:
    """路线分层、组内固定 seed 随机选择；不读取任何任务或效用信息。"""
    candidates = sorted(set(str(vehicle_id) for vehicle_id in vehicle_ids))
    if len(candidates) < count:
        raise ValueError(
            f"候选车辆只有 {len(candidates)} 辆，无法选择 {count} 辆。"
        )
    if count == 0:
        return []

    grouped: dict[str, list[str]] = defaultdict(list)
    for vehicle_id in candidates:
        group = str(route_metadata.get(vehicle_id, {}).get("route_group", "unknown"))
        grouped[group].append(vehicle_id)

    rng = random.Random(seed)
    group_keys = sorted(grouped)
    rng.shuffle(group_keys)

    selected: list[str] = []
    for group in group_keys[: min(count, len(group_keys))]:
        members = sorted(grouped[group])
        selected.append(rng.choice(members))

    if len(selected) < count:
        selected_set = set(selected)
        remaining = [vehicle_id for vehicle_id in candidates if vehicle_id not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])

    return selected


def build_role_assignments(
    complete_ids: list[str],
    route_metadata: dict[str, dict[str, Any]],
    seen_vehicle_ids: set[str],
    config: ScenarioConfig,
) -> tuple[dict[str, Any], bool]:
    """先选 12 辆研究车辆，再在其中固定选择 3 辆 SV。"""
    research = stratified_random_choice(
        complete_ids,
        route_metadata,
        config.num_research_vehicles,
        config.random_seed,
    )

    # 同一个输入和 seed 重复选择，用于本次最小验证中的可复现性断言。
    research_repeat = stratified_random_choice(
        complete_ids,
        route_metadata,
        config.num_research_vehicles,
        config.random_seed,
    )
    if research != research_repeat:
        raise AssertionError("相同 seed 的研究车辆选择不一致。")

    service = stratified_random_choice(
        research,
        route_metadata,
        config.num_service_vehicles,
        config.random_seed + 1,
    )
    service_repeat = stratified_random_choice(
        research,
        route_metadata,
        config.num_service_vehicles,
        config.random_seed + 1,
    )
    if service != service_repeat:
        raise AssertionError("相同 seed 的 SV 选择不一致。")

    service_set = set(service)
    task = [vehicle_id for vehicle_id in research if vehicle_id not in service_set]
    research_set = set(research)
    background = sorted(seen_vehicle_ids - research_set)

    planned_routes = {
        vehicle_id: {
            "route_id": route_metadata[vehicle_id]["route_id"],
            "route_edges": list(route_metadata[vehicle_id]["route_edges"]),
            "route_group": route_metadata[vehicle_id]["route_group"],
            "formal_record_count": route_metadata[vehicle_id]["formal_record_count"],
        }
        for vehicle_id in research
    }

    roles = {
        "research_vehicles": research,
        "service_vehicles": service,
        "task_vehicles": task,
        "background_vehicles": background,
        "seed": config.random_seed,
        "dt": config.dt_s,
        "warmup_time": config.warmup_time_s,
        "episode_time": config.episode_time_s,
        "simulation_end_time": config.simulation_end_s,
        "num_research_vehicles": config.num_research_vehicles,
        "num_service_vehicles": config.num_service_vehicles,
        "num_task_vehicles": config.num_task_vehicles,
        "formal_time_slot_count": len(_formal_step_keys(config)),
        "planned_routes": planned_routes,
    }
    return roles, True


# ============================================================================
# 输出和最小验证
# ============================================================================


def _atomic_write(path: Path, write_callback: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_callback(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_trajectory_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    """写入原始轨迹；不添加 TV/SV 角色列。"""
    ordered_rows = sorted(rows, key=lambda row: (row["time_s"], row["vehicle_id"]))

    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_FIELDS)
            writer.writeheader()
            writer.writerows(ordered_rows)

    _atomic_write(output_file, write)


def write_roles_json(roles: dict[str, Any], output_file: Path) -> None:
    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(roles, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    _atomic_write(output_file, write)


def validate_outputs(
    trajectory_file: Path,
    roles_file: Path,
    complete_ids: list[str],
    config: ScenarioConfig,
) -> dict[str, Any]:
    """对本阶段要求的 9 项最小条件做一次文件级检查。"""
    with trajectory_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_FIELDS:
            raise AssertionError("trajectory.csv 字段与约定不一致。")

        trajectory_ids: set[str] = set()
        row_count = 0
        previous_time = -math.inf
        for row in reader:
            row_count += 1
            vehicle_id = row["vehicle_id"]
            trajectory_ids.add(vehicle_id)
            current_time = float(row["time_s"])
            if current_time < previous_time - 1e-9:
                raise AssertionError("trajectory.csv 的 time_s 不是单调不减。")
            previous_time = current_time

    if row_count == 0:
        raise AssertionError("trajectory.csv 为空。")

    with roles_file.open("r", encoding="utf-8") as stream:
        roles = json.load(stream)

    research = list(roles["research_vehicles"])
    service = list(roles["service_vehicles"])
    task = list(roles["task_vehicles"])
    background = list(roles["background_vehicles"])
    research_set = set(research)
    service_set = set(service)
    task_set = set(task)
    background_set = set(background)

    if len(research) != config.num_research_vehicles:
        raise AssertionError(f"研究车辆数量错误：{len(research)}。")
    if len(service) != config.num_service_vehicles:
        raise AssertionError(f"SV 数量错误：{len(service)}。")
    if len(task) != config.num_task_vehicles:
        raise AssertionError(f"TV 数量错误：{len(task)}。")
    if service_set & task_set:
        raise AssertionError("TV/SV 集合存在交集。")
    if service_set | task_set != research_set:
        raise AssertionError("TV/SV 并集不等于研究车辆集合。")
    if research_set & background_set:
        raise AssertionError("研究车辆和背景车辆存在交集。")
    if research_set | background_set != trajectory_ids:
        raise AssertionError("角色文件没有覆盖 trajectory.csv 中的全部车辆。")
    if not research_set.issubset(set(complete_ids)):
        raise AssertionError("研究车辆中存在正式 episode 轨迹不完整的车辆。")

    all_role_ids = research_set | background_set
    if not all_role_ids.issubset(trajectory_ids):
        raise AssertionError("vehicle_roles.json 中存在 trajectory.csv 找不到的车辆。")

    return {
        "trajectory_rows": row_count,
        "trajectory_vehicle_count": len(trajectory_ids),
        "complete_vehicle_count": len(complete_ids),
        "research_vehicle_count": len(research),
        "service_vehicle_count": len(service),
        "task_vehicle_count": len(task),
        "roles_disjoint": True,
        "seed_selection_reproducible": True,
    }


# ============================================================================
# CLI
# ============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SUMO 完整交通生成、TraCI 轨迹采集和固定 TV/SV 角色划分。"
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_CONFIG.dt_s)
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=DEFAULT_CONFIG.warmup_time_s,
        help="warm-up 时长，单位秒。",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=DEFAULT_CONFIG.episode_time_s,
        help="正式 episode 时长，单位秒。",
    )
    parser.add_argument(
        "--num-research-vehicles",
        type=int,
        default=DEFAULT_CONFIG.num_research_vehicles,
    )
    parser.add_argument(
        "--num-service-vehicles",
        type=int,
        default=DEFAULT_CONFIG.num_service_vehicles,
    )
    parser.add_argument(
        "--num-task-vehicles",
        type=int,
        default=DEFAULT_CONFIG.num_task_vehicles,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_CONFIG.random_seed,
    )
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--sumo-config", default=None)
    parser.add_argument("--net-file", default=None)
    parser.add_argument(
        "--route-file",
        action="append",
        default=[],
        help="显式指定 route/trips 文件；可重复传入。",
    )
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--gui", action="store_true", help="使用 sumo-gui。")
    return parser


def config_from_arguments(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        dt_s=args.dt,
        warmup_time_s=args.warmup_time,
        episode_time_s=args.episode_time,
        num_research_vehicles=args.num_research_vehicles,
        num_service_vehicles=args.num_service_vehicles,
        num_task_vehicles=args.num_task_vehicles,
        random_seed=args.random_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        project_root = find_project_root()
        config = config_from_arguments(args)
        config.validate()

        inventory = discover_sumo_files(project_root)
        print_sumo_inventory(project_root, inventory)

        sumo_home = find_sumo_home(args.sumo_home)
        sumolib, traci = load_sumo_modules(sumo_home)

        source = _choose_existing_source(
            project_root=project_root,
            inventory=inventory,
            explicit_config=args.sumo_config,
            explicit_network=args.net_file,
            explicit_routes=args.route_file,
        )
        output_dir = _resolve_from_project(project_root, args.output_dir)

        if source is None:
            network = _choose_network(project_root, inventory, args.net_file)
            source = generate_random_trips(
                project_root=project_root,
                sumo_home=sumo_home,
                network_file=network,
                output_dir=output_dir,
                config=config,
            )
        else:
            source_type = "randomTrips 生成需求" if source.generated_by_random_trips else "已有需求"
            print(f"复用 {source_type}：")
            print(f"  路网：{source.network_file}")
            print(f"  路线：{', '.join(str(path) for path in source.route_files)}")
            if source.configuration_file:
                print(f"  配置：{source.configuration_file}")

        binary = find_sumo_binary(sumolib, args.gui)
        command = build_sumo_command(binary, source, config)
        rows, route_cache, seen_ids, observed_times = collect_trajectories(
            traci,
            command,
            config,
        )
        if not rows:
            raise RuntimeError("SUMO 运行结束但没有采集到任何车辆轨迹。")

        complete_ids, route_metadata, formal_steps = find_complete_vehicles(
            rows,
            route_cache,
            config,
        )
        print(
            f"采集 {len(rows)} 条轨迹记录，发现 {len(seen_ids)} 辆车辆；"
            f"正式 episode 有 {len(complete_ids)} 辆完整车辆。"
        )
        if len(complete_ids) < config.num_research_vehicles:
            raise RuntimeError(
                f"正式 episode 中完整轨迹车辆只有 {len(complete_ids)} 辆，"
                f"少于要求的 {config.num_research_vehicles} 辆。"
                "请增加 SUMO 车辆量、缩短 episode，或调整路线生成参数。"
            )

        roles, selection_reproducible = build_role_assignments(
            complete_ids,
            route_metadata,
            seen_ids,
            config,
        )
        if not selection_reproducible:
            raise AssertionError("角色选择可复现性验证失败。")

        trajectory_file = output_dir / "trajectory.csv"
        roles_file = output_dir / "vehicle_roles.json"
        write_trajectory_csv(rows, trajectory_file)
        write_roles_json(roles, roles_file)
        validation = validate_outputs(
            trajectory_file,
            roles_file,
            complete_ids,
            config,
        )

        print("最小验证通过：")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        print(f"正式输出：{trajectory_file}")
        print(f"正式输出：{roles_file}")
        print(f"研究车辆：{roles['research_vehicles']}")
        print(f"SV(3)：{roles['service_vehicles']}")
        print(f"TV(9)：{roles['task_vehicles']}")
        print(f"正式时隙数：{len(formal_steps)}；采集时刻数：{len(observed_times)}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
