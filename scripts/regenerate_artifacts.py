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
            "Cloud": float(find_value(config, "nodes.cloud.cpu_cycles_per_s")),
        },
        "prediction": {
            "vehicle_horizon": int(find_value(prediction, "vehicle.horizon")),
            "popularity_history_length": int(
                find_value(popularity_prediction, "history_length")
            ),
            "popularity_hidden_size": int(
                find_value(popularity_prediction, "hidden_size")
            ),
            "scenario_count": int(find_value(scenario, "num_scenarios")),
        },
        "roles_sha256": roles_hash,
    }


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
    """检查节点位置文件的 RSU/UAV ID 是否与当前配置一致。"""

    if not path.is_file():
        return True
    try:
        payload = load_json(path)
        expected = {
            "rsu": {f"RSU_{index}" for index in range(int(find_value(config, "nodes.rsu.count")))},
            "uav": {f"UAV_{index}" for index in range(int(find_value(config, "nodes.uav.count")))},
        }
        return any(set(payload.get(role, {})) != ids for role, ids in expected.items())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
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
    """检查服务数量、服务大小、输入范围和计算密度是否属于当前配置。"""

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
        for row in services:
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


def tasks_are_current(config: dict[str, Any], paths: dict[str, Path]) -> bool:
    """检查 tasks.csv 的 TV 集合、时隙覆盖和服务 ID 范围。"""

    if not paths["tasks"].is_file() or not paths["roles"].is_file():
        return False
    try:
        roles = load_json(paths["roles"])
        tv_ids = {str(value) for value in roles.get("task_vehicles", [])}
        simulation = find_value(config, "simulation", {})
        num_slots = int(round(float(simulation["episode_time_s"]) / float(simulation["dt_s"])))
        service_count = int(find_value(config, "service.count"))
        with paths["tasks"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "slot",
                "vehicle_id",
                "arrival",
                "service_id",
                "input_size",
                "cpu_cycles",
            }
            if not required.issubset(reader.fieldnames or set()):
                return False
            rows = list(reader)
        if len(rows) != len(tv_ids) * num_slots:
            return False
        by_slot: dict[int, set[str]] = {slot: set() for slot in range(num_slots)}
        for row in rows:
            vehicle_id = str(row["vehicle_id"])
            slot = int(row["slot"])
            if vehicle_id not in tv_ids or slot not in by_slot:
                return False
            by_slot[slot].add(vehicle_id)
            if int(row["arrival"]) == 1 and not 0 <= int(row["service_id"]) < service_count:
                return False
        return all(values == tv_ids for values in by_slot.values())
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
    """只用 NumPy/CSV 检查预测场景的服务和车辆维度。"""

    if not all(paths[name].is_file() for name in ("popularity_predictions", "prediction_scenarios", "vehicle_predictions")):
        return False
    try:
        roles = load_json(paths["roles"])
        research_ids = {str(value) for value in roles.get("research_vehicles", [])}
        tv_ids = {str(value) for value in roles.get("task_vehicles", [])}
        sv_ids = {str(value) for value in roles.get("service_vehicles", [])}
        service_count = int(find_value(config, "service.count"))
        prediction_rows: list[dict[str, str]] = []
        with paths["popularity_predictions"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"service_id", "origin_slot", "target_slot", "rho_pred"}.issubset(reader.fieldnames or set()):
                return False
            prediction_rows.extend(reader)
        if not prediction_rows or {int(row["service_id"]) for row in prediction_rows} != set(range(service_count)):
            return False
        with paths["vehicle_predictions"].open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if "vehicle_id" not in (reader.fieldnames or set()):
                return False
            vehicle_ids = {str(row["vehicle_id"]) for row in reader}
        if vehicle_ids != research_ids:
            return False
        import numpy as np

        loaded = np.load(paths["prediction_scenarios"], allow_pickle=False)
        try:
            metadata = json.loads(str(loaded["metadata_json"].item()))
            if int(metadata["F"]) != service_count:
                return False
            if set(loaded["vehicle_ids"].astype(str)) != research_ids:
                return False
            if set(loaded["task_vehicle_ids"].astype(str)) != tv_ids:
                return False
            if set(loaded["service_vehicle_ids"].astype(str)) != sv_ids:
                return False
            if loaded["rho_hat"].shape[-1] != service_count:
                return False
            if loaded["arrival"].shape[-1] != len(tv_ids):
                return False
            return True
        finally:
            loaded.close()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ImportError):
        return False


def run_task_generation() -> None:
    """调用现有任务生成器，重生成 service/popularity/tasks/TV cache。"""

    result = task_scenario.main([])
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


def run_prediction_generation(project_root: Path) -> None:
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
            [sys.executable, str(script)],
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
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": "system_config.toml",
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
    return parser


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

        force_regeneration = bool(args.force)
        role_change = force_regeneration or roles_need_regeneration(config, paths["roles"])
        node_change = force_regeneration or node_positions_need_regeneration(config, paths["node_positions"])
        catalog_change = force_regeneration or not catalog_is_current(config, paths["catalog"])
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
        )["task"]
        task_signature_change = (
            not no_previous_metadata
            and previous_signature.get("task") != task_signature
        )
        service_change = catalog_change
        task_change = (
            force_regeneration
            or service_change
            or role_change
            or not tasks_are_current(config, paths)
            or task_signature_change
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
        formal_config = project_config.load_config()
        del formal_config

        if task_change:
            run_task_generation()
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

        prediction_change = (
            force_regeneration
            or role_change
            or service_change
            or task_change
            or not prediction_artifacts_are_current(config, paths)
        )
        if prediction_change:
            run_prediction_generation(PROJECT_ROOT)
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
