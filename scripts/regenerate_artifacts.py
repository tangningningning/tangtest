#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 system_config.toml 的依赖关系重生成并校验正式 artifact。

脚本只编排已有场景生成器，不实现新的任务、缓存、预测、通信或优化算法。
车辆角色复用已有 trajectory.csv，不启动 SUMO；预测阶段只调用现有的 GRU
预测场景生成流程。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:
    # 兼容 Python 3.10；优先使用环境中的 tomli，再使用 pip 自带副本。
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "system_config.toml"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/generated/generation_metadata.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports/artifact_consistency_report.md"
FLOAT_TOLERANCE = 1.0e-9
EXPECTED_CACHE_ROLES = ("TV", "SV", "RSU", "UAV")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import project_config
from scenario import sumo_scenario, task_scenario


def find_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """读取点号路径配置；路径不存在时返回给定默认值。"""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """把配置中的相对路径解析为项目内的绝对路径。"""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_raw_config(config_path: Path) -> dict[str, Any]:
    """读取未经 artifact 校验的 TOML，支持先修复上游 artifact。"""

    with config_path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise ValueError("system_config.toml 根对象必须是 table。")
    return value


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON object 文件。"""

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是 object：{path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """以临时文件原子写入 JSON，并在结束时清理临时文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    """计算文件哈希，用于记录生成输入而不复制文件内容。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_hash_mismatch(
    metadata: dict[str, Any],
    artifact_name: str,
    path: Path,
) -> bool:
    """若 metadata 已记录哈希，则检测 artifact 是否被外部修改。"""

    recorded = metadata.get("artifact_sha256", {})
    if not isinstance(recorded, dict) or artifact_name not in recorded:
        return False
    return not path.is_file() or str(recorded[artifact_name]) != sha256_file(path)


def validate_input_config(config: dict[str, Any]) -> None:
    """检查本次 baseline 的规模、任务范围和缓存比例关系。"""

    simulation = find_value(config, "simulation", {})
    research = int(find_value(simulation, "num_research_vehicles", 0))
    service_vehicles = int(find_value(simulation, "num_service_vehicles", 0))
    task_vehicles = int(find_value(simulation, "num_task_vehicles", 0))
    if service_vehicles + task_vehicles != research:
        raise ValueError("车辆数量不满足 SV + TV = research vehicles。")

    for key in ("nodes.rsu.count", "nodes.uav.count", "service.count"):
        if int(find_value(config, key, 0)) < 1:
            raise ValueError(f"{key} 必须是正整数。")

    generation = find_value(config, "service.generation", {})
    range_pairs = (
        ("service_size_min_gb", "service_size_max_gb"),
        ("input_size_min_kb", "input_size_max_kb"),
        ("cycles_per_bit_min", "cycles_per_bit_max"),
    )
    for lower_key, upper_key in range_pairs:
        lower = float(generation[lower_key])
        upper = float(generation[upper_key])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0:
            raise ValueError(f"service.generation.{lower_key} 范围非法。")
        if upper < lower:
            raise ValueError(f"service.generation.{upper_key} 小于下界。")

    policy = find_value(config, "cache.capacity_policy", {})
    ratios = {
        role: float(policy[f"{role.lower()}_ratio"])
        for role in EXPECTED_CACHE_ROLES
    }
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in ratios.values()):
        raise ValueError("cache.capacity_policy 中的容量比例必须位于 (0, 1)。")
    if not (
        ratios["RSU"] > ratios["UAV"] > ratios["SV"] > ratios["TV"]
    ):
        raise ValueError("容量比例必须满足 RSU > UAV > SV > TV。")


def current_signature(
    config: dict[str, Any],
    capacities_mb: dict[str, float],
    roles_hash: str | None = None,
) -> dict[str, Any]:
    """提取用于增量判断的正式配置快照。"""

    simulation = find_value(config, "simulation", {})
    generation = find_value(config, "service.generation", {})
    task = find_value(config, "task", {})
    policy = find_value(config, "cache.capacity_policy", {})
    prediction = find_value(config, "prediction", {})
    popularity_prediction = find_value(prediction, "popularity", {})
    scenario = find_value(config, "scenario", {})
    return {
        "vehicles": {
            "research": int(simulation["num_research_vehicles"]),
            "SV": int(simulation["num_service_vehicles"]),
            "TV": int(simulation["num_task_vehicles"]),
            "sumo_seed": int(simulation["sumo_seed"]),
        },
        "infrastructure": {
            "RSU": int(find_value(config, "nodes.rsu.count")),
            "UAV": int(find_value(config, "nodes.uav.count")),
        },
        "service_count": int(find_value(config, "service.count")),
        "service_generation": {
            key: float(generation[key])
            for key in (
                "service_size_min_gb",
                "service_size_max_gb",
                "input_size_min_kb",
                "input_size_max_kb",
                "cycles_per_bit_min",
                "cycles_per_bit_max",
            )
        },
        "task": {
            key: task[key]
            for key in (
                "arrival_probability",
                "popularity_zeta",
                "popularity_epoch_length",
                "popularity_changes_per_epoch",
                "input_relative_jitter",
                "load_relative_jitter",
                "master_seed",
            )
        },
        "time": {
            "dt_s": float(simulation["dt_s"]),
            "warmup_time_s": float(simulation["warmup_time_s"]),
            "episode_time_s": float(simulation["episode_time_s"]),
        },
        "cache_policy": {
            role: float(policy[f"{role.lower()}_ratio"])
            for role in EXPECTED_CACHE_ROLES
        },
        "cache_capacity_mb": {
            role: float(capacities_mb[role]) for role in EXPECTED_CACHE_ROLES
        },
        "compute": {
            "TV": float(find_value(config, "nodes.vehicle.local_cpu_cycles_per_s")),
            "SV": float(find_value(config, "nodes.sv.cpu_cycles_per_s")),
            "RSU": float(find_value(config, "nodes.rsu.cpu_cycles_per_s")),
            "UAV": float(find_value(config, "nodes.uav.cpu_cycles_per_s")),
            "Cloud": float(find_value(config, "nodes.cloud.total_cpu_cycles_per_s")),
        },
        "prediction": {
            "vehicle": json.loads(json.dumps(find_value(prediction, "vehicle", {}))),
            "popularity": json.loads(json.dumps(popularity_prediction)),
            "scenario": json.loads(json.dumps(scenario)),
        },
        "node_placement": {
            "sumo_net": str(find_value(config, "paths.sumo_net")),
            "rsu_altitude_m": float(find_value(config, "nodes.rsu.deployment_altitude_m")),
            "uav_altitude_m": float(find_value(config, "nodes.uav.deployment_altitude_m")),
            "uav_offset_fraction": float(
                find_value(config, "nodes.uav.deployment_offset_fraction")
            ),
        },
        "roles_sha256": roles_hash,
    }


def prediction_signature_matches(
    previous: Any,
    current: dict[str, Any],
) -> bool:
    """比较预测配置，并兼容旧 metadata 的四字段签名。"""

    if previous == current:
        return True
    if not isinstance(previous, dict):
        return False
    if {"vehicle", "popularity", "scenario"}.issubset(previous):
        return False
    return (
        int(previous.get("vehicle_horizon", -1))
        == int(find_value(current, "vehicle.horizon", -2))
        and int(previous.get("popularity_history_length", -1))
        == int(find_value(current, "popularity.history_length", -2))
        and int(previous.get("popularity_hidden_size", -1))
        == int(find_value(current, "popularity.hidden_size", -2))
        and int(previous.get("scenario_count", -1))
        == int(find_value(current, "scenario.num_scenarios", -2))
    )


def artifact_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """从统一配置解析所有本次流程使用的 artifact 路径。"""

    return {
        "roles": resolve_path(project_root, find_value(config, "paths.vehicle_roles")),
        "trajectory": resolve_path(project_root, find_value(config, "paths.trajectory")),
        "catalog": resolve_path(project_root, find_value(config, "paths.service_catalog")),
        "popularity": resolve_path(project_root, find_value(config, "paths.popularity")),
        "popularity_predictions": resolve_path(
            project_root,
            Path(find_value(config, "paths.popularity")).parent
            / "popularity_predictions.csv",
        ),
        "popularity_history": resolve_path(
            project_root,
            find_value(config, "paths.popularity_history"),
        ),
        "tasks": resolve_path(project_root, find_value(config, "paths.tasks")),
        "tv_local_cache": resolve_path(
            project_root,
            find_value(config, "paths.tv_local_cache"),
        ),
        "vehicle_predictions": resolve_path(
            project_root,
            find_value(config, "paths.vehicle_predictions"),
        ),
        "prediction_scenarios": resolve_path(
            project_root,
            find_value(config, "paths.prediction_scenarios"),
        ),
        "popularity_gru": resolve_path(
            project_root,
            find_value(config, "paths.popularity_gru"),
        ),
        "node_positions": resolve_path(
            project_root,
            find_value(config, "paths.node_positions"),
        ),
    }


def load_roles_counts(path: Path) -> dict[str, int]:
    """读取角色文件中的研究、SV、TV 数量。"""

    roles = load_json(path)
    return {
        "research": len(roles.get("research_vehicles", [])),
        "SV": len(roles.get("service_vehicles", [])),
        "TV": len(roles.get("task_vehicles", [])),
    }


def roles_need_regeneration(config: dict[str, Any], path: Path) -> bool:
    """判断角色 artifact 是否需要基于既有轨迹重新筛选。"""

    if not path.is_file():
        return True
    try:
        roles = load_json(path)
        simulation = find_value(config, "simulation", {})
        expected = {
            "research": int(simulation["num_research_vehicles"]),
            "SV": int(simulation["num_service_vehicles"]),
            "TV": int(simulation["num_task_vehicles"]),
        }
        actual = load_roles_counts(path)
        return (
            actual != expected
            or int(roles.get("seed", -1)) != int(simulation["sumo_seed"])
            or not math.isclose(
                float(roles.get("dt", -1.0)),
                float(simulation["dt_s"]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            or not math.isclose(
                float(roles.get("warmup_time", -1.0)),
                float(simulation["warmup_time_s"]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            or not math.isclose(
                float(roles.get("episode_time", -1.0)),
                float(simulation["episode_time_s"]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True


def build_role_config(config: dict[str, Any]) -> sumo_scenario.ScenarioConfig:
    """从原始 TOML 构造角色筛选所需的场景配置。"""

    simulation = find_value(config, "simulation", {})
    return sumo_scenario.ScenarioConfig(
        dt_s=float(simulation["dt_s"]),
        warmup_time_s=float(simulation["warmup_time_s"]),
        episode_time_s=float(simulation["episode_time_s"]),
        num_research_vehicles=int(simulation["num_research_vehicles"]),
        num_service_vehicles=int(simulation["num_service_vehicles"]),
        num_task_vehicles=int(simulation["num_task_vehicles"]),
        random_seed=int(simulation["sumo_seed"]),
        traffic_seed=int(simulation.get("traffic_seed", simulation["sumo_seed"])),
        random_trips_period_s=float(simulation.get("random_trips_period_s", 1.0)),
        random_trips_departure_begin_s=float(
            simulation.get("random_trips_departure_begin_s", 0.0)
        ),
        random_trips_departure_end_s=float(
            simulation.get("random_trips_departure_end_s", 60.0)
        ),
        random_trips_min_distance_m=float(
            simulation.get("random_trips_min_distance_m", 0.0)
        ),
        random_trips_intermediate=int(simulation.get("random_trips_intermediate", 0)),
        random_trips_maxtries=int(simulation.get("random_trips_maxtries", 1)),
    )


def regenerate_roles(config: dict[str, Any], paths: dict[str, Path]) -> None:
    """复用既有正式轨迹重建角色文件，不运行 SUMO 或 TraCI。"""

    role_config = build_role_config(config)
    role_config.validate()
    sumo_scenario.reuse_existing_trajectory_roles(
        paths["trajectory"],
        paths["roles"],
        role_config,
    )


def node_positions_need_regeneration(config: dict[str, Any], path: Path) -> bool:
    """按当前路网和部署配置重算期望结果，检查完整节点位置 artifact。"""

    if not path.is_file():
        return True
    try:
        from scenario import node_placement

        payload = load_json(path)
        net_file = resolve_path(PROJECT_ROOT, find_value(config, "paths.sumo_net"))
        network, boundary = node_placement.read_sumo_network(net_file)
        candidates = node_placement.extract_junction_candidates(network)
        selected = node_placement.select_rsu_candidates(
            candidates,
            int(find_value(config, "nodes.rsu.count")),
        )
        uav_positions = node_placement.generate_uav_positions(
            selected,
            int(find_value(config, "nodes.uav.count")),
            boundary,
            float(find_value(config, "nodes.uav.deployment_offset_fraction")),
            float(find_value(config, "nodes.uav.deployment_altitude_m")),
        )
        expected = node_placement.build_node_positions_payload(
            net_file,
            PROJECT_ROOT,
            boundary,
            selected,
            uav_positions,
            float(find_value(config, "nodes.rsu.deployment_altitude_m")),
        )
        return payload != expected
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ModuleNotFoundError):
        return True


def regenerate_nodes(config_path: Path) -> None:
    """调用现有道路感知节点部署脚本生成节点位置。"""

    from scenario import node_placement

    result = node_placement.main(["--config", str(config_path)])
    if result != 0:
        raise RuntimeError("node_placement.py 生成失败。")


def deterministic_catalog_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """按正式 seed 在内存中预生成 catalog，用于计算容量而不写临时 JSON。"""

    task_config = task_scenario.task_config_from_project_config(config)
    task_config.validate(int(find_value(config, "simulation.num_task_vehicles")))
    master_seed = int(find_value(config, "task.master_seed"))
    rngs, _ = task_scenario.make_rngs(master_seed, task_config.rng_stream_names)
    return task_scenario.make_service_catalog(task_config, rngs["service_catalog"])


def catalog_is_current(config: dict[str, Any], path: Path) -> bool:
    """按当前 seed 重建期望服务目录，避免仅凭“落在范围内”误判为最新。"""

    if not path.is_file():
        return False
    try:
        catalog = load_json(path)
        generation = find_value(config, "service.generation", {})
        services = catalog.get("services")
        if int(catalog.get("service_count", -1)) != int(find_value(config, "service.count")):
            return False
        if not isinstance(services, list) or len(services) != int(find_value(config, "service.count")):
            return False
        expected_ids = set(range(int(find_value(config, "service.count"))))
        if {int(row["service_id"]) for row in services} != expected_ids:
            return False
        expected_services = sorted(
            deterministic_catalog_from_config(config),
            key=lambda row: int(row["service_id"]),
        )
        actual_services = sorted(services, key=lambda row: int(row["service_id"]))
        for row, expected_row in zip(actual_services, expected_services):
            raw_size_gb = row.get("service_size_gb")
            if raw_size_gb is None:
                raw_size_gb = row.get("service_size")
            size_gb = float(raw_size_gb)
            input_kb = float(row.get("input_size_kb"))
            program_mb = float(row["program_size"])
            mean_input_bit = float(row["mean_input_size"])
            cycles = float(row["cycles_per_bit"])
            if not (
                float(generation["service_size_min_gb"]) <= size_gb <= float(generation["service_size_max_gb"])
                and float(generation["input_size_min_kb"]) <= input_kb <= float(generation["input_size_max_kb"])
                and float(generation["cycles_per_bit_min"]) <= cycles <= float(generation["cycles_per_bit_max"])
                and math.isclose(
                    program_mb,
                    size_gb * task_scenario.SERVICE_SIZE_MB_PER_GB,
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                and math.isclose(
                    mean_input_bit,
                    input_kb * task_scenario.INPUT_SIZE_BITS_PER_KB,
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            ):
                return False
            for key, expected_value in expected_row.items():
                actual_value = row.get(key)
                if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                    if actual_value is None or not math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=1.0e-12,
                        abs_tol=FLOAT_TOLERANCE,
                    ):
                        return False
                elif actual_value != expected_value:
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def catalog_total_size_mb(config: dict[str, Any], path: Path) -> float:
    """读取或预生成服务目录并返回运行时使用的 MB 总大小。"""

    if catalog_is_current(config, path):
        catalog = load_json(path)
        return float(sum(float(row["program_size"]) for row in catalog["services"]))
    catalog = deterministic_catalog_from_config(config)
    return float(sum(float(row["program_size"]) for row in catalog))


def calculate_capacity_targets(
    config: dict[str, Any],
    total_size_mb: float,
) -> dict[str, float]:
    """按 service catalog 总大小和配置比例计算四类节点容量。"""

    if not math.isfinite(total_size_mb) or total_size_mb <= 0.0:
        raise ValueError("service catalog 总大小必须是有限正数。")
    policy = find_value(config, "cache.capacity_policy", {})
    capacities = {
        role: total_size_mb * float(policy[f"{role.lower()}_ratio"])
        for role in EXPECTED_CACHE_ROLES
    }
    if not (capacities["RSU"] > capacities["UAV"] > capacities["SV"] > capacities["TV"]):
        raise AssertionError("自动计算的缓存容量未满足 RSU > UAV > SV > TV。")
    if any(value >= total_size_mb for value in capacities.values()):
        raise AssertionError("单个节点缓存容量不能覆盖整个 service catalog。")
    return capacities


def render_number(value: float) -> str:
    """以稳定的十进制形式写入 TOML 数值。"""

    return format(float(value), ".15g")


def replace_toml_value(
    text: str,
    section: str,
    key: str,
    value: float,
) -> str:
    """只替换指定 TOML section 的数值，保留其余注释和格式。"""

    lines = text.splitlines(keepends=True)
    current_section: str | None = None
    replaced = False
    assignment = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s+#.*)?$")
    for index, line in enumerate(lines):
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body, newline = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, newline = line[:-1], "\n"
        section_match = re.match(r"^\s*\[([^\]]+)\]\s*$", body)
        if section_match:
            current_section = section_match.group(1)
            continue
        if current_section != section:
            continue
        match = assignment.match(body)
        if match:
            comment = match.group(3) or ""
            lines[index] = f"{match.group(1)}{render_number(value)}{comment}{newline}"
            replaced = True
            break
    if not replaced:
        raise ValueError(f"TOML 中找不到 [{section}] {key}。")
    return "".join(lines)


def update_capacity_config(
    config_path: Path,
    config: dict[str, Any],
    capacities_mb: dict[str, float],
) -> bool:
    """把按 catalog 计算出的容量写回统一配置入口。"""

    current = {
        "TV": float(find_value(config, "nodes.vehicle.local_cache_capacity_mb")),
        "SV": float(find_value(config, "nodes.sv.cache_capacity_mb")),
        "RSU": float(find_value(config, "nodes.rsu.cache_capacity_mb")),
        "UAV": float(find_value(config, "nodes.uav.cache_capacity_mb")),
    }
    changed = any(
        not math.isclose(current[role], capacities_mb[role], rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
        for role in EXPECTED_CACHE_ROLES
    )
    if not changed:
        return False

    text = config_path.read_text(encoding="utf-8")
    replacements = (
        ("nodes.vehicle", "local_cache_capacity_mb", "TV"),
        ("nodes.sv", "cache_capacity_mb", "SV"),
        ("nodes.rsu", "cache_capacity_mb", "RSU"),
        ("nodes.uav", "cache_capacity_mb", "UAV"),
    )
    for section, key, role in replacements:
        text = replace_toml_value(text, section, key, capacities_mb[role])
    temporary = config_path.with_name(config_path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(config_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def deterministic_task_artifacts_from_config(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """使用当前正式 seed 在内存中重建任务链，不写入任何 artifact。"""

    roles = load_json(paths["roles"])
    task_vehicles = [str(value) for value in roles.get("task_vehicles", [])]
    task_config = task_scenario.task_config_from_project_config(config)
    task_config.validate(len(task_vehicles))
    simulation = find_value(config, "simulation", {})
    return task_scenario.build_artifacts(
        task_vehicles,
        float(simulation["dt_s"]),
        float(simulation["warmup_time_s"]),
        float(simulation["episode_time_s"]),
        int(find_value(config, "task.master_seed")),
        task_config,
    )


def tasks_are_current(config: dict[str, Any], paths: dict[str, Path]) -> bool:
    """逐行比较当前 seed 的确定性期望任务，覆盖全部任务生成参数。"""

    if not paths["tasks"].is_file() or not paths["roles"].is_file():
        return False
    try:
        expected_rows = deterministic_task_artifacts_from_config(config, paths)[
            "task_rows"
        ]
        with paths["tasks"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != task_scenario.TASK_FIELDS:
                return False
            rows = list(reader)
        if len(rows) != len(expected_rows):
            return False
        integer_fields = {"slot", "arrival", "service_id"}
        text_fields = {"vehicle_id"}
        for row, expected in zip(rows, expected_rows):
            for field in task_scenario.TASK_FIELDS:
                actual_text = "" if row[field] is None else str(row[field])
                expected_text = "" if expected[field] is None else str(expected[field])
                if actual_text == "" or expected_text == "":
                    equal = actual_text == expected_text
                elif field in text_fields:
                    equal = str(row[field]) == str(expected[field])
                elif field in integer_fields:
                    equal = int(row[field]) == int(expected[field])
                else:
                    equal = math.isclose(
                        float(row[field]),
                        float(expected[field]),
                        rel_tol=1.0e-12,
                        abs_tol=FLOAT_TOLERANCE,
                    )
                if not equal:
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, csv.Error):
        return False


def cache_is_current(config: dict[str, Any], paths: dict[str, Path]) -> bool:
    """检查 TV 初始缓存集合和容量约束。"""

    if not paths["tv_local_cache"].is_file():
        return False
    try:
        catalog = load_json(paths["catalog"])
        roles = load_json(paths["roles"])
        capacity = float(find_value(config, "nodes.vehicle.local_cache_capacity_mb"))
        validation = task_scenario.validate_tv_local_cache(
            load_json(paths["tv_local_cache"]),
            [str(value) for value in roles.get("task_vehicles", [])],
            catalog,
            capacity,
        )
        return bool(
            validation["all_used_capacity_within_limit"]
            and int(validation["service_count"]) == int(find_value(config, "service.count"))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prediction_artifacts_are_current(config: dict[str, Any], paths: dict[str, Path]) -> bool:
    """检查预测 CSV/NPZ 的完整维度、时间轴、seed 与模型元数据。"""

    if not all(paths[name].is_file() for name in ("popularity_predictions", "prediction_scenarios", "vehicle_predictions")):
        return False
    try:
        roles = load_json(paths["roles"])
        research_ids = {str(value) for value in roles.get("research_vehicles", [])}
        tv_ids = {str(value) for value in roles.get("task_vehicles", [])}
        sv_ids = {str(value) for value in roles.get("service_vehicles", [])}
        service_count = int(find_value(config, "service.count"))
        simulation = find_value(config, "simulation", {})
        num_slots = int(
            round(float(simulation["episode_time_s"]) / float(simulation["dt_s"]))
        )
        horizon = int(find_value(config, "prediction.vehicle.horizon"))
        scenario_count = int(find_value(config, "scenario.num_scenarios"))
        prediction_rows: list[dict[str, str]] = []
        with paths["popularity_predictions"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"service_id", "origin_slot", "target_slot", "rho_pred"}.issubset(reader.fieldnames or set()):
                return False
            prediction_rows.extend(reader)
        if len(prediction_rows) != num_slots * horizon * service_count:
            return False
        popularity_keys = {
            (int(row["origin_slot"]), int(row["target_slot"]), int(row["service_id"]))
            for row in prediction_rows
        }
        expected_popularity_keys = {
            (origin, origin + step, service_id)
            for origin in range(num_slots)
            for step in range(horizon)
            for service_id in range(service_count)
        }
        if popularity_keys != expected_popularity_keys:
            return False
        with paths["vehicle_predictions"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required_vehicle_fields = {
                "vehicle_id",
                "origin_slot",
                "observation_cutoff_slot",
                "horizon",
                "target_slot",
                "x_pred",
                "y_pred",
            }
            if not required_vehicle_fields.issubset(reader.fieldnames or set()):
                return False
            vehicle_rows = list(reader)
        if len(vehicle_rows) != num_slots * horizon * len(research_ids):
            return False
        vehicle_keys = {
            (str(row["vehicle_id"]), int(row["origin_slot"]), int(row["horizon"]))
            for row in vehicle_rows
        }
        expected_vehicle_keys = {
            (vehicle_id, origin, step)
            for vehicle_id in research_ids
            for origin in range(num_slots)
            for step in range(horizon)
        }
        if vehicle_keys != expected_vehicle_keys:
            return False
        if any(
            int(row["target_slot"]) != int(row["origin_slot"]) + int(row["horizon"])
            or int(row["observation_cutoff_slot"]) != int(row["origin_slot"]) - 1
            for row in vehicle_rows
        ):
            return False
        import numpy as np

        loaded = np.load(paths["prediction_scenarios"], allow_pickle=False)
        try:
            metadata = json.loads(str(loaded["metadata_json"].item()))
            if (
                int(metadata["F"]) != service_count
                or int(metadata["S"]) != scenario_count
                or int(metadata["H_p"]) != horizon
                or int(metadata["scenario_seed"]) != int(find_value(config, "scenario.master_seed"))
                or not math.isclose(
                    float(metadata["lambda"]),
                    float(find_value(config, "task.arrival_probability")),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                or int(metadata["formal_time_slot_count"]) != num_slots
                or int(metadata["max_resample_attempts"])
                != int(find_value(config, "scenario.max_resample_attempts"))
            ):
                return False
            if set(loaded["vehicle_ids"].astype(str)) != research_ids:
                return False
            if set(loaded["task_vehicle_ids"].astype(str)) != tv_ids:
                return False
            if set(loaded["service_vehicle_ids"].astype(str)) != sv_ids:
                return False
            expected_position_shape = (
                num_slots,
                scenario_count,
                horizon,
                len(research_ids),
                2,
            )
            expected_task_shape = (
                num_slots,
                scenario_count,
                horizon,
                len(tv_ids),
            )
            if loaded["positions_xy"].shape != expected_position_shape:
                return False
            if any(
                loaded[name].shape != expected_task_shape
                for name in ("arrival", "service_id", "input_size", "cpu_cycles")
            ):
                return False
            if loaded["rho_hat"].shape != (num_slots, horizon, service_count):
                return False
            if loaded["origin_slots"].astype(int).tolist() != list(range(num_slots)):
                return False
            if loaded["scenario_ids"].astype(int).tolist() != list(range(scenario_count)):
                return False
            if loaded["horizons"].astype(int).tolist() != list(range(horizon)):
                return False
            noise = metadata.get("noise_related_metadata_reference", {})
            if (
                not math.isclose(
                    float(noise.get("sigma_position_m", float("nan"))),
                    float(find_value(config, "prediction.vehicle.sigma_position_m")),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                or not math.isclose(
                    float(noise.get("sigma_speed_mps", float("nan"))),
                    float(find_value(config, "prediction.vehicle.sigma_speed_mps")),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
                or int(noise.get("observation_noise_seed", -1))
                != int(find_value(config, "prediction.vehicle.observation_noise_seed"))
            ):
                return False
            gru = metadata.get("GRU checkpoint metadata reference", {})
            if (
                int(gru.get("H_p", -1)) != horizon
                or int(gru.get("L_rho", -1))
                != int(find_value(config, "prediction.popularity.history_length"))
                or int(gru.get("hidden_size", -1))
                != int(find_value(config, "prediction.popularity.hidden_size"))
            ):
                return False
            return True
        finally:
            loaded.close()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ImportError):
        return False


def run_task_generation(config_path: Path) -> None:
    """调用现有任务生成器，重生成 service/popularity/tasks/TV cache。"""

    result = task_scenario.main(["--config", str(config_path)])
    if result != 0:
        raise RuntimeError("task_scenario.py 生成失败。")


def regenerate_tv_cache(config: dict[str, Any], paths: dict[str, Path]) -> None:
    """只按当前 service catalog 和 TV 容量重生成本地缓存初始状态。"""

    roles = load_json(paths["roles"])
    catalog = load_json(paths["catalog"])
    capacity = float(find_value(config, "nodes.vehicle.local_cache_capacity_mb"))
    tv_ids = [str(value) for value in roles.get("task_vehicles", [])]
    cache = task_scenario.generate_tv_local_cache(
        tv_ids,
        catalog,
        capacity,
        int(find_value(config, "experiment.seed")),
    )
    task_scenario.validate_tv_local_cache(cache, tv_ids, catalog, capacity)
    write_json_atomic(paths["tv_local_cache"], cache)


def run_prediction_generation(project_root: Path, config_path: Path) -> None:
    """调用现有 GRU 和预测场景生成器，不启动 SUMO 或 PPO。"""

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "预测 artifact 需要当前 Python 环境安装 PyTorch；"
            "未修改旧预测 artifact。"
        )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    import_paths = [str(project_root / "src")]
    tomli_spec = importlib.util.find_spec("tomli")
    if tomli_spec is not None and tomli_spec.origin:
        import_paths.append(str(Path(tomli_spec.origin).parent.parent))
    import_paths.extend(
        item for item in existing_pythonpath.split(os.pathsep) if item
    )
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    for script_name in ("popularity_prediction.py", "scenario_generator.py"):
        script = project_root / "src/scenario" / script_name
        subprocess.run(
            [sys.executable, str(script), "--config", str(config_path)],
            cwd=project_root,
            env=environment,
            check=True,
        )


def write_report(
    report_path: Path,
    config: dict[str, Any],
    capacities_mb: dict[str, float],
    total_size_mb: float,
    stages: dict[str, str],
    checks: dict[str, bool],
    errors: list[str],
) -> None:
    """写入 artifact 依赖执行和一致性检查报告。"""

    generation = find_value(config, "service.generation", {})
    lines = [
        "# Artifact 一致性报告",
        "",
        "本报告由 `scripts/regenerate_artifacts.py` 根据 `system_config.toml` 生成。",
        "角色重建复用既有 `trajectory.csv`，未启动 SUMO；未运行 PPO 或完整实验。",
        "",
        "## 当前配置摘要",
        "",
        f"- 车辆：research={find_value(config, 'simulation.num_research_vehicles')}，"
        f"SV={find_value(config, 'simulation.num_service_vehicles')}，"
        f"TV={find_value(config, 'simulation.num_task_vehicles')}",
        f"- 基础设施：RSU={find_value(config, 'nodes.rsu.count')}，"
        f"UAV={find_value(config, 'nodes.uav.count')}",
        f"- service.count={find_value(config, 'service.count')}",
        f"- input size 基准范围={generation['input_size_min_kb']}--{generation['input_size_max_kb']} KB",
        f"- cycles_per_bit={generation['cycles_per_bit_min']}--{generation['cycles_per_bit_max']}",
        f"- service size={generation['service_size_min_gb']}--{generation['service_size_max_gb']} GB",
        "",
        "## 服务目录与缓存容量",
        "",
        f"- service catalog 总大小：{total_size_mb:.6f} MB",
        "",
        "| 节点 | 容量 (MB) | catalog 比例 |",
        "|---|---:|---:|",
    ]
    policy = find_value(config, "cache.capacity_policy", {})
    for role in EXPECTED_CACHE_ROLES:
        lines.append(f"| {role} | {capacities_mb[role]:.6f} | {float(policy[f'{role.lower()}_ratio']):.2%} |")
    lines.extend(["", "## 执行阶段", "", "| 阶段 | 状态 |", "|---|---|"])
    for name, status in stages.items():
        lines.append(f"| {name} | {status} |")
    lines.extend(["", "## 一致性检查", "", "| 检查项 | 结果 |", "|---|---|"])
    for name, passed in checks.items():
        lines.append(f"| {name} | {'通过' if passed else '失败'} |")
    if errors:
        lines.extend(["", "## 错误或未完成项", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(["", "## 依赖关系", "", "- service 参数 -> service_catalog、popularity、tasks、TV 初始缓存、预测场景", "- vehicle 参数/角色集合 -> vehicle_roles、tasks、TV 初始缓存、vehicle_predictions、prediction_scenarios", "- task 参数 -> popularity、tasks、GRU 预测输入和 prediction_scenarios", "- cache 容量比例 -> system_config.toml 中的派生 MB 容量和 TV 初始缓存", "- RSU/UAV 数量 -> node_positions.json；位置仍由 node_placement.py 道路感知生成", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_metadata(
    config: dict[str, Any],
    paths: dict[str, Path],
    capacities_mb: dict[str, float],
    total_size_mb: float,
    stages: dict[str, str],
    config_path: Path,
) -> dict[str, Any]:
    """组装当前 artifact 的正式元数据。"""

    generation = find_value(config, "service.generation", {})
    roles_hash = sha256_file(paths["roles"])
    signature = current_signature(config, capacities_mb, roles_hash)
    relative_paths = {
        name: str(path.relative_to(PROJECT_ROOT).as_posix())
        for name, path in paths.items()
        if path.exists()
    }
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(
            config_path.relative_to(PROJECT_ROOT).as_posix()
            if config_path.is_relative_to(PROJECT_ROOT)
            else config_path
        ),
        "config_sha256": sha256_file(config_path),
        "vehicles": {
            "research": signature["vehicles"]["research"],
            "SV": signature["vehicles"]["SV"],
            "TV": signature["vehicles"]["TV"],
        },
        "RSU": signature["infrastructure"]["RSU"],
        "UAV": signature["infrastructure"]["UAV"],
        "service_count": signature["service_count"],
        "task_input_range": [
            float(generation["input_size_min_kb"]),
            float(generation["input_size_max_kb"]),
        ],
        "cycles_per_bit": [
            float(generation["cycles_per_bit_min"]),
            float(generation["cycles_per_bit_max"]),
        ],
        "service_size_gb": [
            float(generation["service_size_min_gb"]),
            float(generation["service_size_max_gb"]),
        ],
        "cache_capacity": {role: float(capacities_mb[role]) for role in EXPECTED_CACHE_ROLES},
        "service_catalog_total_mb": total_size_mb,
        "configuration": signature,
        "artifacts": relative_paths,
        "artifact_sha256": artifact_hashes,
        "stages": stages,
        "vehicle_predictions_depend_on": "trajectory.csv + current vehicle_roles.json",
        "prediction_note": "预测 artifact 只在车辆/服务/任务/预测维度变化时重生成。",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造统一 artifact 入口参数解析器。"""

    parser = argparse.ArgumentParser(
        description="根据 system_config.toml 增量重生成并校验正式 artifact。"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="统一 TOML 配置路径。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 generation_metadata.json，重生成全部依赖 artifact。",
    )
    parser.add_argument(
        "--allow-predictor-training",
        action="store_true",
        help="显式允许在预测 artifact 失效时重新训练 popularity GRU。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只读检查配置、metadata 和 artifact，不写入或重新生成文件。",
    )
    return parser


def read_only_consistency_checks(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float], float]:
    """执行统一入口的只读一致性检查，不修改配置和 artifact。"""

    total_size_mb = catalog_total_size_mb(config, paths["catalog"])
    capacities_mb = calculate_capacity_targets(config, total_size_mb)
    configured_capacities = {
        "TV": float(find_value(config, "nodes.vehicle.local_cache_capacity_mb")),
        "SV": float(find_value(config, "nodes.sv.cache_capacity_mb")),
        "RSU": float(find_value(config, "nodes.rsu.cache_capacity_mb")),
        "UAV": float(find_value(config, "nodes.uav.cache_capacity_mb")),
    }
    expected_roles = {
        "research": int(find_value(config, "simulation.num_research_vehicles")),
        "SV": int(find_value(config, "simulation.num_service_vehicles")),
        "TV": int(find_value(config, "simulation.num_task_vehicles")),
    }
    roles_hash = sha256_file(paths["roles"]) if paths["roles"].is_file() else None
    expected_signature = current_signature(config, capacities_mb, roles_hash)
    recorded_hashes = metadata.get("artifact_sha256", {})
    artifact_hashes_match = isinstance(recorded_hashes, dict) and all(
        path.is_file()
        and name in recorded_hashes
        and str(recorded_hashes[name]) == sha256_file(path)
        for name, path in paths.items()
    )
    config_source = (
        str(config_path.relative_to(PROJECT_ROOT).as_posix())
        if config_path.is_relative_to(PROJECT_ROOT)
        else str(config_path)
    )
    checks = {
        "正式配置可解析": True,
        "派生缓存容量与配置一致": all(
            math.isclose(
                configured_capacities[role],
                capacities_mb[role],
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            )
            for role in EXPECTED_CACHE_ROLES
        ),
        "vehicle_roles 与配置一致": (
            paths["roles"].is_file()
            and load_roles_counts(paths["roles"]) == expected_roles
        ),
        "service_catalog 与配置和 seed 一致": catalog_is_current(
            config,
            paths["catalog"],
        ),
        "tasks.csv 与当前角色和任务配置一致": tasks_are_current(config, paths),
        "TV 初始缓存与容量约束一致": cache_is_current(config, paths),
        "node_positions 与路网和节点配置一致": not node_positions_need_regeneration(
            config,
            paths["node_positions"],
        ),
        "预测 artifact 维度和元数据一致": prediction_artifacts_are_current(
            config,
            paths,
        ),
        "generation metadata 配置签名一致": (
            metadata.get("configuration") == expected_signature
        ),
        "generation metadata 配置文件哈希一致": (
            metadata.get("config_source") == config_source
            and metadata.get("config_sha256") == sha256_file(config_path)
        ),
        "generation metadata artifact 哈希完整且一致": artifact_hashes_match,
    }
    return checks, capacities_mb, total_size_mb


def main(argv: list[str] | None = None) -> int:
    """执行配置读取、增量生成、元数据写入和一致性报告。"""

    args = build_parser().parse_args(argv)
    config_path = resolve_path(PROJECT_ROOT, args.config)
    metadata_path = PROJECT_ROOT / "data/generated/generation_metadata.json"
    report_path = DEFAULT_REPORT_PATH
    stages: dict[str, str] = {}
    checks: dict[str, bool] = {}
    errors: list[str] = []
    capacities_mb = {role: 0.0 for role in EXPECTED_CACHE_ROLES}
    total_size_mb = 0.0
    config: dict[str, Any] = {}
    paths: dict[str, Path] = {}

    try:
        config = load_raw_config(config_path)
        validate_input_config(config)
        paths = artifact_paths(config, PROJECT_ROOT)
        previous = load_json(metadata_path) if metadata_path.is_file() else {}

        if bool(args.check_only):
            # 先调用正式加载器，确保节点实例、参数状态和唯一配置入口均可通过。
            project_config.load_config(config_path)
            checks, capacities_mb, total_size_mb = read_only_consistency_checks(
                config_path,
                config,
                paths,
                previous,
            )
            print(
                json.dumps(
                    {
                        "mode": "check_only",
                        "writes_performed": 0,
                        "checks": checks,
                        "all_passed": all(checks.values()),
                        "service_catalog_total_mb": total_size_mb,
                        "derived_cache_capacity_mb": capacities_mb,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0 if all(checks.values()) else 1

        force_regeneration = bool(args.force)
        role_change = (
            force_regeneration
            or roles_need_regeneration(config, paths["roles"])
            or recorded_hash_mismatch(previous, "roles", paths["roles"])
            or recorded_hash_mismatch(previous, "trajectory", paths["trajectory"])
        )
        node_change = (
            force_regeneration
            or node_positions_need_regeneration(config, paths["node_positions"])
            or recorded_hash_mismatch(previous, "node_positions", paths["node_positions"])
        )
        catalog_change = (
            force_regeneration
            or not catalog_is_current(config, paths["catalog"])
            or recorded_hash_mismatch(previous, "catalog", paths["catalog"])
        )
        previous_signature = previous.get("configuration")
        no_previous_metadata = not isinstance(previous_signature, dict)
        if no_previous_metadata:
            previous_signature = {}
        task_signature = current_signature(
            config,
            {
                "TV": float(find_value(config, "nodes.vehicle.local_cache_capacity_mb")),
                "SV": float(find_value(config, "nodes.sv.cache_capacity_mb")),
                "RSU": float(find_value(config, "nodes.rsu.cache_capacity_mb")),
                "UAV": float(find_value(config, "nodes.uav.cache_capacity_mb")),
            },
        )
        task_signature_change = (
            not no_previous_metadata
            and previous_signature.get("task") != task_signature["task"]
        )
        prediction_signature_change = (
            not no_previous_metadata
            and not prediction_signature_matches(
                previous_signature.get("prediction"),
                task_signature["prediction"],
            )
        )
        service_change = catalog_change
        task_change = (
            force_regeneration
            or service_change
            or role_change
            or not tasks_are_current(config, paths)
            or recorded_hash_mismatch(previous, "tasks", paths["tasks"])
            or task_signature_change
        )

        prediction_change = (
            force_regeneration
            or role_change
            or service_change
            or task_change
            or prediction_signature_change
            or not prediction_artifacts_are_current(config, paths)
            or any(
                recorded_hash_mismatch(previous, name, paths[name])
                for name in (
                    "popularity_predictions",
                    "vehicle_predictions",
                    "prediction_scenarios",
                    "popularity_gru",
                )
            )
        )
        if prediction_change and not bool(args.allow_predictor_training):
            raise RuntimeError(
                "预测 artifact 已失效，但训练保护处于开启状态；"
                "未写入任何 artifact。确认需要重新训练 popularity GRU 后，"
                "显式添加 --allow-predictor-training。"
            )

        total_size_mb = catalog_total_size_mb(config, paths["catalog"])
        capacities_mb = calculate_capacity_targets(config, total_size_mb)
        capacity_config_changed = update_capacity_config(config_path, config, capacities_mb)
        if capacity_config_changed:
            config = load_raw_config(config_path)
            validate_input_config(config)
            paths = artifact_paths(config, PROJECT_ROOT)
        cache_change = (
            service_change
            or role_change
            or capacity_config_changed
            or force_regeneration
            or not cache_is_current(config, paths)
            or recorded_hash_mismatch(
                previous,
                "tv_local_cache",
                paths["tv_local_cache"],
            )
        )

        if role_change:
            regenerate_roles(config, paths)
            stages["vehicle_roles"] = "已复用 trajectory.csv 重生成"
        else:
            stages["vehicle_roles"] = "跳过（与配置一致）"

        if node_change:
            regenerate_nodes(config_path)
            stages["node_positions"] = "已调用 node_placement.py 重生成"
        else:
            stages["node_positions"] = "跳过（RSU/UAV 数量一致）"

        # 角色和节点先就绪后，正式配置校验才能安全检查所有入口。
        formal_config = project_config.load_config(config_path)
        del formal_config

        if task_change:
            run_task_generation(config_path)
            stages["service_catalog"] = "由 task_scenario.py 重生成"
            stages["tasks"] = "由 task_scenario.py 重生成"
            stages["tv_local_cache"] = "由 task_scenario.py 重生成"
            stages["popularity"] = "由 task_scenario.py 重生成"
        else:
            stages["service_catalog"] = "跳过（与 metadata 一致）"
            stages["tasks"] = "跳过（与 metadata 一致）"
            stages["tv_local_cache"] = "跳过（与 metadata 一致）"
            stages["popularity"] = "跳过（与 metadata 一致）"
        if cache_change and not task_change:
            regenerate_tv_cache(config, paths)
            stages["tv_local_cache"] = "按新容量重生成"

        if prediction_change:
            run_prediction_generation(PROJECT_ROOT, config_path)
            stages["prediction"] = "GRU 与 prediction_scenarios 重生成"
        else:
            stages["prediction"] = "跳过（与 metadata 一致）"

        config = load_raw_config(config_path)
        paths = artifact_paths(config, PROJECT_ROOT)
        total_size_mb = catalog_total_size_mb(config, paths["catalog"])
        capacities_mb = calculate_capacity_targets(config, total_size_mb)
        checks = {
            "system_config 可解析且车辆约束成立": True,
            "vehicle_roles 数量与配置一致": load_roles_counts(paths["roles"])
            == {
                "research": int(find_value(config, "simulation.num_research_vehicles")),
                "SV": int(find_value(config, "simulation.num_service_vehicles")),
                "TV": int(find_value(config, "simulation.num_task_vehicles")),
            },
            "service_catalog 数量与范围一致": catalog_is_current(config, paths["catalog"]),
            "tasks.csv TV 集合和时隙覆盖一致": tasks_are_current(config, paths),
            "TV 初始缓存容量约束成立": cache_is_current(config, paths),
            "RSU/UAV 节点数量一致": not node_positions_need_regeneration(config, paths["node_positions"]),
            "预测 artifact 维度一致": prediction_artifacts_are_current(config, paths),
        }
        if not all(checks.values()):
            errors.extend(name for name, passed in checks.items() if not passed)
            raise RuntimeError("artifact 一致性检查未全部通过。")

        metadata = build_metadata(
            config,
            paths,
            capacities_mb,
            total_size_mb,
            stages,
            config_path,
        )
        write_json_atomic(metadata_path, metadata)
        write_report(
            report_path,
            config,
            capacities_mb,
            total_size_mb,
            stages,
            checks,
            errors,
        )
        print("artifact 重新生成与一致性检查完成。")
        print(f"metadata: {metadata_path}")
        print(f"report: {report_path}")
        return 0
    except Exception as exc:
        errors.append(str(exc))
        if config:
            try:
                write_report(
                    report_path,
                    config,
                    capacities_mb,
                    total_size_mb,
                    stages,
                    checks,
                    errors,
                )
            except Exception as report_error:
                print(f"一致性报告写入失败：{report_error}", file=sys.stderr)
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
