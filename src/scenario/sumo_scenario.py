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

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scenario.realization import derive_episode_seeds
except ModuleNotFoundError:  # direct ``python src/scenario/sumo_scenario.py``
    from realization import derive_episode_seeds

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
    num_research_vehicles: int = 25
    num_service_vehicles: int = 5
    num_task_vehicles: int = 20
    random_seed: int = 4401
    traffic_seed: int = 4401
    research_vehicle_role_seed: int | None = None
    service_vehicle_role_seed: int | None = None

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


def _copy_bytes_to_canonical_file(source_file: Path, destination_file: Path) -> None:
    """Copy generated bytes into the canonical directory before finalization.

    ``Path.replace`` preserves the ACL attached to a file created in an
    unrelated temporary directory on Windows.  The canonical destination is
    therefore created by the parent directory itself, while the temporary
    file is used only as a byte source.
    """

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    with source_file.open("rb") as source_stream, destination_file.open(
        "wb"
    ) as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())








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












def _managed_demand_metadata_path(route_file: Path) -> Path:
    return route_file.with_name("sumo_demand.meta.json")


def _format_inventory_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


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
        generated_hash = _sha256_file(generated_route)
        _copy_bytes_to_canonical_file(generated_route, route_file)
        if _sha256_file(route_file) != generated_hash:
            raise RuntimeError("randomTrips.py canonical route finalization hash mismatch。")

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
    research_seed = (
        config.random_seed
        if config.research_vehicle_role_seed is None
        else int(config.research_vehicle_role_seed)
    )
    service_seed = (
        config.random_seed + 1
        if config.service_vehicle_role_seed is None
        else int(config.service_vehicle_role_seed)
    )
    research = fixed_seed_random_choice(
        complete_ids,
        config.num_research_vehicles,
        research_seed,
    )

    # 同一个输入和 seed 重复选择，用于本次最小验证中的可复现性断言。
    research_repeat = fixed_seed_random_choice(
        complete_ids,
        config.num_research_vehicles,
        research_seed,
    )
    if research != research_repeat:
        raise AssertionError("相同 seed 的研究车辆选择不一致。")

    service = fixed_seed_random_choice(
        research,
        config.num_service_vehicles,
        service_seed,
    )
    service_repeat = fixed_seed_random_choice(
        research,
        config.num_service_vehicles,
        service_seed,
    )
    if service != service_repeat:
        raise AssertionError("相同 seed 的 SV 选择不一致。")

    service_set = set(service)
    task = [vehicle_id for vehicle_id in research if vehicle_id not in service_set]
    # Preserve the original seeded research order and role-specific selection
    # orders. Scenario axes are built from SV followed by TV and carry their IDs.
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
        "research_vehicles": list(research),
        "service_vehicles": list(service),
        "task_vehicles": list(task),
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
        "research_selection_seed": research_seed,
        "service_selection_seed": service_seed,
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




# ============================================================================
# CLI
# ============================================================================
