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
import hashlib
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
    num_research_vehicles: int = 30
    num_service_vehicles: int = 5
    num_task_vehicles: int = 25
    random_seed: int = 4401
    traffic_seed: int = 4401

    # 仅在 data/ 下没有可直接运行的交通需求时使用官方 randomTrips.py。
    random_trips_period_s: float = 1.0
    random_trips_departure_begin_s: float = 0.0
    random_trips_departure_end_s: float = 60.0
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
        if self.random_trips_departure_begin_s < 0.0:
            raise ValueError("randomTrips departure begin 不能为负数。")
        if self.random_trips_departure_end_s <= self.random_trips_departure_begin_s:
            raise ValueError("randomTrips departure end 必须大于 begin。")
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
        Path(__file__).resolve().parents[2],
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _managed_demand_metadata_path(route_file: Path) -> Path:
    return route_file.with_name("sumo_demand.meta.json")


def _network_metadata_path(project_root: Path, network_file: Path) -> str:
    return _format_inventory_path(project_root, network_file)


def _expected_random_trips_metadata(
    project_root: Path,
    network_file: Path,
    config: ScenarioConfig,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": "randomTrips",
        "network": {
            "path": _network_metadata_path(project_root, network_file),
            "sha256": _sha256_file(network_file),
        },
        "random_seed": int(config.random_seed),
        "traffic_seed": int(config.traffic_seed),
        "period_s": float(config.random_trips_period_s),
        "departure_begin_s": float(config.random_trips_departure_begin_s),
        "departure_end_s": float(config.random_trips_departure_end_s),
        "min_distance_m": float(config.random_trips_min_distance_m),
        "intermediate": int(config.random_trips_intermediate),
        "maxtries": int(config.random_trips_maxtries),
        "vehicle_class": "passenger",
    }


def _load_demand_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _managed_demand_matches(
    project_root: Path,
    route_file: Path,
    network_file: Path,
    config: ScenarioConfig,
) -> bool:
    if not _is_readable_file(route_file):
        return False
    metadata = _load_demand_metadata(_managed_demand_metadata_path(route_file))
    expected = _expected_random_trips_metadata(project_root, network_file, config)
    if metadata is None:
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


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
    managed_route = (
        project_root / "data" / "generated" / "sumo_demand.rou.xml"
    ).resolve(strict=False)
    for config_file in inventory["configurations"]:
        source = _source_from_config(config_file)
        if source is not None and managed_route not in source.route_files:
            return source

    network = _choose_network(project_root, inventory, None)
    generated_dir = (project_root / "data" / "generated").resolve(strict=False)

    # 已有交通需求优先于本模块之前生成的文件；排序保证跨平台稳定。
    candidates = [
        path
        for path in inventory["route_files"]
        if path.parent != generated_dir and path != managed_route
    ] + [
        path
        for path in inventory["route_files"]
        if path.parent == generated_dir and path != managed_route
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
    metadata_file = _managed_demand_metadata_path(route_file)

    if route_file.exists():
        if not _is_readable_file(route_file):
            raise PermissionError(f"已有生成需求文件不可读，不覆盖它：{route_file}")
        if _managed_demand_matches(project_root, route_file, network_file, config):
            print(f"复用本模块之前生成的交通需求：{route_file}")
            return TrafficSource(
                network_file=network_file,
                route_files=(route_file,),
                generated_by_random_trips=True,
            )
        print("已有 randomTrips 需求的 metadata 与当前配置不匹配，将重新生成。")

    random_trips = sumo_home / "tools" / "randomTrips.py"
    if not random_trips.is_file():
        raise FileNotFoundError(f"SUMO 官方 randomTrips.py 不存在：{random_trips}")

    command = [
        sys.executable,
        str(random_trips),
        "-n",
        str(network_file),
        "-o",
        "{trip_file}",
        "-r",
        "{route_file}",
        "-b",
        f"{config.random_trips_departure_begin_s:.9g}",
        "-e",
        f"{config.random_trips_departure_end_s:.9g}",
        "-p",
        f"{config.random_trips_period_s:.9g}",
        "-s",
        str(config.traffic_seed),
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
        generated_route = Path(temporary) / "sumo_demand.rou.xml"
        command = [
            str(trip_file)
            if item == "{trip_file}"
            else str(generated_route)
            if item == "{route_file}"
            else item
            for item in command
        ]
        subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=True,
        )
        if not _is_readable_file(generated_route):
            raise RuntimeError(
                "randomTrips.py 执行结束但没有生成可读临时路线文件。"
            )
        generated_route.replace(route_file)

    if not _is_readable_file(route_file):
        raise RuntimeError(
            f"randomTrips.py 执行结束但没有生成可读路线文件：{route_file}"
        )

    metadata = _expected_random_trips_metadata(project_root, network_file, config)

    def write_metadata(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    _atomic_write(metadata_file, write_metadata)

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
# 正式区间完整轨迹筛选和固定 seed 随机角色选择
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


def load_existing_trajectory(
    trajectory_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[str], list[float]]:
    """读取已有 SUMO 轨迹并恢复筛选所需的 route metadata。

    该入口只复用已经落盘的真实 SUMO 记录，不启动 SUMO、不生成车辆，
    用于在系统规模变化时重新执行正式的完整轨迹过滤和角色划分。
    """

    numeric_fields = (
        "time_s",
        "x_m",
        "y_m",
        "speed_mps",
        "acceleration_mps2",
        "lane_position_m",
        "distance_m",
    )
    rows: list[dict[str, Any]] = []
    route_cache: dict[str, dict[str, Any]] = {}
    seen_vehicle_ids: set[str] = set()
    observed_times: list[float] = []
    previous_time = -math.inf
    with trajectory_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_FIELDS:
            raise ValueError("已有 trajectory.csv 字段与正式约定不一致。")
        for raw_row in reader:
            row = dict(raw_row)
            for field in numeric_fields:
                row[field] = float(row[field])
            vehicle_id = str(row["vehicle_id"])
            route_edges = tuple(
                edge for edge in str(row["route_edges"]).split() if edge
            )
            route = {
                "route_id": str(row["route_id"]),
                "route_edges": route_edges,
            }
            if vehicle_id in route_cache and route_cache[vehicle_id] != route:
                raise ValueError(
                    f"vehicle {vehicle_id} 的已有轨迹 route metadata 不一致。"
                )
            route_cache[vehicle_id] = route
            time_s = float(row["time_s"])
            if time_s < previous_time - 1.0e-9:
                raise ValueError("已有 trajectory.csv 的 time_s 不是单调不减。")
            if not observed_times or not math.isclose(
                time_s, observed_times[-1], rel_tol=0.0, abs_tol=1.0e-9
            ):
                observed_times.append(time_s)
            previous_time = time_s
            seen_vehicle_ids.add(vehicle_id)
            rows.append(row)
    if not rows:
        raise ValueError("已有 trajectory.csv 为空。")
    return rows, route_cache, seen_vehicle_ids, observed_times


def find_complete_vehicles(
    rows: Iterable[dict[str, Any]],
    route_cache: dict[str, dict[str, Any]],
    config: ScenarioConfig,
) -> tuple[list[str], dict[str, dict[str, Any]], set[int]]:
    """保留正式区间完整且有至少一条 warm-up 历史记录的车辆。"""
    formal_steps = _formal_step_keys(config)
    formal_start = min(formal_steps)
    rows_by_vehicle: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    preformal_by_vehicle: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        step = _step_key(float(row["time_s"]), config.dt_s)
        vehicle_id = str(row["vehicle_id"])
        if step < formal_start and _valid_trajectory_row(row):
            preformal_by_vehicle[vehicle_id].add(step)
        if step not in formal_steps:
            continue
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
        if formal_start > 0 and not preformal_by_vehicle.get(vehicle_id):
            # origin=0 的 posterior cutoff 是 formal_start-1；没有 warm-up
            # 观测的车辆不能进入正式 research 集合。
            continue

        complete_ids.append(vehicle_id)
        route = route_cache.get(vehicle_id, {})
        route_id = str(route.get("route_id", ""))
        route_edges = tuple(str(edge) for edge in route.get("route_edges", ()))
        metadata[vehicle_id] = {
            "route_id": route_id,
            "route_edges": route_edges,
            "formal_record_count": len(time_rows),
            "preformal_record_count": len(preformal_by_vehicle.get(vehicle_id, set())),
        }

    return complete_ids, metadata, formal_steps


def fixed_seed_random_choice(
    vehicle_ids: Iterable[str],
    count: int,
    seed: int,
) -> list[str]:
    """从完整候选集合中固定 seed 随机抽样；不读取 route 分组、任务或效用。"""
    candidates = sorted(set(str(vehicle_id) for vehicle_id in vehicle_ids))
    if len(candidates) < count:
        raise ValueError(
            f"候选车辆只有 {len(candidates)} 辆，无法选择 {count} 辆。"
        )
    if count == 0:
        return []
    rng = random.Random(seed)
    return rng.sample(candidates, count)


def build_role_assignments(
    complete_ids: list[str],
    route_metadata: dict[str, dict[str, Any]],
    seen_vehicle_ids: set[str],
    config: ScenarioConfig,
) -> tuple[dict[str, Any], bool]:
    """先从完整候选中固定 seed 抽 research，再从 research 抽 SV。"""
    research = fixed_seed_random_choice(
        complete_ids,
        config.num_research_vehicles,
        config.random_seed,
    )

    # 同一个输入和 seed 重复选择，用于本次最小验证中的可复现性断言。
    research_repeat = fixed_seed_random_choice(
        complete_ids,
        config.num_research_vehicles,
        config.random_seed,
    )
    if research != research_repeat:
        raise AssertionError("相同 seed 的研究车辆选择不一致。")

    service = fixed_seed_random_choice(
        research,
        config.num_service_vehicles,
        config.random_seed + 1,
    )
    service_repeat = fixed_seed_random_choice(
        research,
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
            "formal_record_count": route_metadata[vehicle_id]["formal_record_count"],
            "preformal_record_count": route_metadata[vehicle_id]["preformal_record_count"],
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
        "selection_method": "fixed_seed_random_choice",
        "research_selection_seed": config.random_seed,
        "service_selection_seed": config.random_seed + 1,
        "complete_candidate_count": len(complete_ids),
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
        formal_rows_by_vehicle: dict[str, dict[int, int]] = defaultdict(dict)
        for row in reader:
            row_count += 1
            vehicle_id = row["vehicle_id"]
            trajectory_ids.add(vehicle_id)
            current_time = float(row["time_s"])
            if current_time < previous_time - 1e-9:
                raise AssertionError("trajectory.csv 的 time_s 不是单调不减。")
            previous_time = current_time
            step = _step_key(current_time, config.dt_s)
            if step in _formal_step_keys(config):
                per_vehicle = formal_rows_by_vehicle.setdefault(vehicle_id, {})
                per_vehicle[step] = per_vehicle.get(step, 0) + 1

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
    formal_steps = _formal_step_keys(config)
    for vehicle_id in research:
        per_vehicle = formal_rows_by_vehicle.get(vehicle_id, {})
        if set(per_vehicle) != formal_steps or any(
            count != 1 for count in per_vehicle.values()
        ):
            raise AssertionError(
                f"research vehicle {vehicle_id} 未覆盖正式 slot 0...{len(formal_steps) - 1}。"
            )

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
        "formal_time_slot_count": len(formal_steps),
        "research_formal_records_exactly_once": True,
        "research_has_preformal_history": all(
            int(roles["planned_routes"][vehicle_id].get("preformal_record_count", 0)) > 0
            for vehicle_id in research
        ),
    }


def reuse_existing_trajectory_roles(
    trajectory_file: Path,
    roles_file: Path,
    config: ScenarioConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[float]]:
    """基于已有真实 SUMO trajectory 重新筛选并写入动态角色文件。"""

    rows, route_cache, seen_ids, observed_times = load_existing_trajectory(
        trajectory_file
    )
    complete_ids, route_metadata, formal_steps = find_complete_vehicles(
        rows,
        route_cache,
        config,
    )
    if len(complete_ids) < config.num_research_vehicles:
        raise RuntimeError(
            f"已有 trajectory.csv 中完整研究车辆只有 {len(complete_ids)} 辆，"
            f"少于要求的 {config.num_research_vehicles} 辆。"
        )
    roles, selection_reproducible = build_role_assignments(
        complete_ids,
        route_metadata,
        seen_ids,
        config,
    )
    if not selection_reproducible:
        raise AssertionError("角色选择可复现性验证失败。")
    write_roles_json(roles, roles_file)
    validation = validate_outputs(trajectory_file, roles_file, complete_ids, config)
    validation["trajectory_reused_without_sumo"] = True
    validation["formal_step_count"] = len(formal_steps)
    return roles, validation, complete_ids, observed_times


# ============================================================================
# CLI
# ============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SUMO 完整交通生成、TraCI 轨迹采集和固定 TV/SV 角色划分。"
    )
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=None,
        help="warm-up 时长，单位秒。",
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=None,
        help="正式 episode 时长，单位秒。",
    )
    parser.add_argument(
        "--num-research-vehicles",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-service-vehicles",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-task-vehicles",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
    )
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--departure-begin", type=float, default=None)
    parser.add_argument("--departure-end", type=float, default=None)
    parser.add_argument("--random-trips-period", type=float, default=None)
    parser.add_argument("--random-trips-min-distance", type=float, default=None)
    parser.add_argument("--random-trips-intermediate", type=int, default=None)
    parser.add_argument("--random-trips-maxtries", type=int, default=None)
    parser.add_argument("--sumo-home", default=None)
    parser.add_argument("--sumo-config", default=None)
    parser.add_argument("--net-file", default=None)
    parser.add_argument(
        "--trajectory-file",
        default=None,
        help="复用已有真实 SUMO trajectory.csv 时使用的输入文件。",
    )
    parser.add_argument(
        "--reuse-existing-trajectory",
        action="store_true",
        help="只重用已有真实轨迹并重新划分角色，不启动 SUMO。",
    )
    parser.add_argument(
        "--route-file",
        action="append",
        default=[],
        help="显式指定 route/trips 文件；可重复传入。",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gui", action="store_true", help="使用 sumo-gui。")
    return parser


def config_from_arguments(
    args: argparse.Namespace,
    project_config: dict[str, Any] | None = None,
) -> ScenarioConfig:
    if project_config is None:
        base = DEFAULT_CONFIG
    else:
        from common import project_config as project_config_module

        get_value = project_config_module.get_config_value
        base = ScenarioConfig(
            dt_s=float(get_value(project_config, "simulation.dt_s", DEFAULT_CONFIG.dt_s)),
            warmup_time_s=float(
                get_value(
                    project_config,
                    "simulation.warmup_time_s",
                    DEFAULT_CONFIG.warmup_time_s,
                )
            ),
            episode_time_s=float(
                get_value(
                    project_config,
                    "simulation.episode_time_s",
                    DEFAULT_CONFIG.episode_time_s,
                )
            ),
            num_research_vehicles=int(
                get_value(
                    project_config,
                    "simulation.num_research_vehicles",
                    DEFAULT_CONFIG.num_research_vehicles,
                )
            ),
            num_service_vehicles=int(
                get_value(
                    project_config,
                    "simulation.num_service_vehicles",
                    DEFAULT_CONFIG.num_service_vehicles,
                )
            ),
            num_task_vehicles=int(
                get_value(
                    project_config,
                    "simulation.num_task_vehicles",
                    DEFAULT_CONFIG.num_task_vehicles,
                )
            ),
            random_seed=int(
                get_value(
                    project_config,
                    "simulation.sumo_seed",
                    DEFAULT_CONFIG.random_seed,
                )
            ),
            traffic_seed=int(
                get_value(
                    project_config,
                    "simulation.traffic_seed",
                    DEFAULT_CONFIG.traffic_seed,
                )
            ),
            random_trips_period_s=float(
                get_value(
                    project_config,
                    "simulation.random_trips_period_s",
                    DEFAULT_CONFIG.random_trips_period_s,
                )
            ),
            random_trips_departure_begin_s=float(
                get_value(
                    project_config,
                    "simulation.random_trips_departure_begin_s",
                    DEFAULT_CONFIG.random_trips_departure_begin_s,
                )
            ),
            random_trips_departure_end_s=float(
                get_value(
                    project_config,
                    "simulation.random_trips_departure_end_s",
                    DEFAULT_CONFIG.random_trips_departure_end_s,
                )
            ),
            random_trips_min_distance_m=float(
                get_value(
                    project_config,
                    "simulation.random_trips_min_distance_m",
                    DEFAULT_CONFIG.random_trips_min_distance_m,
                )
            ),
            random_trips_intermediate=int(
                get_value(
                    project_config,
                    "simulation.random_trips_intermediate",
                    DEFAULT_CONFIG.random_trips_intermediate,
                )
            ),
            random_trips_maxtries=int(
                get_value(
                    project_config,
                    "simulation.random_trips_maxtries",
                    DEFAULT_CONFIG.random_trips_maxtries,
                )
            ),
        )

    def choose(name: str, value: Any) -> Any:
        return getattr(base, name) if value is None else value

    return ScenarioConfig(
        dt_s=choose("dt_s", args.dt),
        warmup_time_s=choose("warmup_time_s", args.warmup_time),
        episode_time_s=choose("episode_time_s", args.episode_time),
        num_research_vehicles=choose(
            "num_research_vehicles", args.num_research_vehicles
        ),
        num_service_vehicles=choose(
            "num_service_vehicles", args.num_service_vehicles
        ),
        num_task_vehicles=choose("num_task_vehicles", args.num_task_vehicles),
        random_seed=choose("random_seed", args.random_seed),
        traffic_seed=choose("traffic_seed", args.traffic_seed),
        random_trips_period_s=choose(
            "random_trips_period_s", args.random_trips_period
        ),
        random_trips_departure_begin_s=choose(
            "random_trips_departure_begin_s", args.departure_begin
        ),
        random_trips_departure_end_s=choose(
            "random_trips_departure_end_s", args.departure_end
        ),
        random_trips_min_distance_m=choose(
            "random_trips_min_distance_m", args.random_trips_min_distance
        ),
        random_trips_intermediate=choose(
            "random_trips_intermediate", args.random_trips_intermediate
        ),
        random_trips_maxtries=choose(
            "random_trips_maxtries", args.random_trips_maxtries
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        project_root = find_project_root()
        from common import project_config as project_config_module

        formal_config = project_config_module.load_config()
        config = config_from_arguments(args, formal_config)
        config.validate()

        output_dir = _resolve_from_project(
            project_root,
            args.output_dir
            or Path(
                project_config_module.get_config_value(
                    formal_config,
                    "paths.vehicle_roles",
                    "data/generated/vehicle_roles.json",
                )
            ).parent,
        )

        if args.reuse_existing_trajectory:
            trajectory_file = _resolve_from_project(
                project_root,
                args.trajectory_file
                or project_config_module.get_config_value(
                    formal_config,
                    "paths.trajectory",
                    "data/generated/trajectory.csv",
                ),
            )
            roles_file = output_dir / "vehicle_roles.json"
            roles, validation, complete_ids, observed_times = reuse_existing_trajectory_roles(
                trajectory_file,
                roles_file,
                config,
            )
            print("复用已有真实 SUMO trajectory，未启动 SUMO：")
            print(f"  trajectory: {trajectory_file}")
            print(f"  complete candidates: {len(complete_ids)}")
            print(f"  observed times: {len(observed_times)}")
            print("最小验证通过：")
            for key, value in validation.items():
                print(f"  {key}: {value}")
            print(f"正式输出：{roles_file}")
            print(f"研究车辆({len(roles['research_vehicles'])})：{roles['research_vehicles']}")
            print(f"SV({len(roles['service_vehicles'])})：{roles['service_vehicles']}")
            print(f"TV({len(roles['task_vehicles'])})：{roles['task_vehicles']}")
            return 0

        inventory = discover_sumo_files(project_root)
        print_sumo_inventory(project_root, inventory)

        sumo_home = find_sumo_home(args.sumo_home)
        sumolib, traci = load_sumo_modules(sumo_home)

        source = _choose_existing_source(
            project_root=project_root,
            inventory=inventory,
            explicit_config=args.sumo_config,
            explicit_network=args.net_file
            or project_config_module.get_config_value(
                formal_config,
                "paths.sumo_net",
            ),
            explicit_routes=args.route_file,
        )
        if source is None:
            network = _choose_network(
                project_root,
                inventory,
                args.net_file
                or project_config_module.get_config_value(
                    formal_config,
                    "paths.sumo_net",
                ),
            )
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
        print(f"SV({len(roles['service_vehicles'])})：{roles['service_vehicles']}")
        print(f"TV({len(roles['task_vehicles'])})：{roles['task_vehicles']}")
        print(f"正式时隙数：{len(formal_steps)}；采集时刻数：{len(observed_times)}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
