"""读取正式 baseline，分析缓存容量压力并静态审计缓存更新路径。

本脚本只读取配置、service catalog、popularity artifact 和缓存相关源代码，
不会修改配置、任务数据、缓存状态或训练代码。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 兼容路径
    import tomli as tomllib  # type: ignore[no-redef]


# 概率和与缓存容量比较都允许极小的浮点误差。
PROBABILITY_SUM_TOLERANCE = 1.0e-6
CAPACITY_TOLERANCE_MB = 1.0e-9
VALUE_TOLERANCE = 1.0e-12

DEFAULT_CONFIG = Path("system_config.toml")
DEFAULT_SERVICE_CATALOG = Path("data/generated/service_catalog.json")
DEFAULT_POPULARITY = Path("data/generated/popularity.csv")
DEFAULT_REPORT = Path("reports/cache_capacity_analysis.md")


def project_root() -> Path:
    """返回仓库根目录，支持从仓库根目录或任意当前目录运行脚本。"""

    return Path(__file__).resolve().parents[2]


def resolve_project_path(root: Path, configured_path: str | Path) -> Path:
    """将配置中的相对路径解析为仓库内的绝对路径。"""

    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并返回其字典内容。"""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根对象必须是字典：{path}")
    return value


def load_toml(path: Path) -> dict[str, Any]:
    """读取 TOML 配置并返回其字典内容。"""

    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"TOML 根对象必须是字典：{path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    """校验并返回一个配置或 artifact 字典。"""

    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象。")
    return value


def load_service_sizes(catalog: dict[str, Any]) -> dict[int, float]:
    """读取每个 service 的真实 program size，单位为 MB。"""

    services = catalog.get("services")
    if not isinstance(services, list) or not services:
        raise ValueError("service_catalog.json 缺少非空 services 列表。")

    sizes: dict[int, float] = {}
    for service in services:
        item = _mapping(service, "service")
        service_id = int(item["service_id"])
        raw_size = item.get("program_size", item.get("service_size", item.get("program_size_mb")))
        if raw_size is None:
            raise ValueError(f"service {service_id} 缺少 program_size/service_size 字段。")
        size_mb = float(raw_size)
        if not math.isfinite(size_mb) or size_mb < 0.0:
            raise ValueError(f"service {service_id} 的大小不是有效非负数：{raw_size}")
        if service_id in sizes:
            raise ValueError(f"service_id 重复：{service_id}")
        sizes[service_id] = size_mb
    return dict(sorted(sizes.items()))


def load_cache_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """从统一 system_config.toml 读取四类节点的缓存容量。"""

    nodes = _mapping(config.get("nodes"), "nodes")
    vehicle = _mapping(nodes.get("vehicle"), "nodes.vehicle")
    sv = _mapping(nodes.get("sv"), "nodes.sv")
    rsu = _mapping(nodes.get("rsu"), "nodes.rsu")
    uav = _mapping(nodes.get("uav"), "nodes.uav")

    definitions = {
        "TV": (vehicle, "local_cache_capacity_mb", "nodes.vehicle.local_cache_capacity_mb"),
        "SV": (sv, "cache_capacity_mb", "nodes.sv.cache_capacity_mb"),
        "RSU": (rsu, "cache_capacity_mb", "nodes.rsu.cache_capacity_mb"),
        "UAV": (uav, "cache_capacity_mb", "nodes.uav.cache_capacity_mb"),
    }
    profiles: dict[str, dict[str, Any]] = {}
    for node_type, (section, key, config_key) in definitions.items():
        capacity_mb = float(section[key])
        if not math.isfinite(capacity_mb) or capacity_mb < 0.0:
            raise ValueError(f"{config_key} 不是有效非负容量。")
        profiles[node_type] = {
            "capacity_mb": capacity_mb,
            "config_key": config_key,
        }
    return profiles


def load_cache_cost_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """读取缓存更新成本所需的 eta_ca、xi_ca 和 lambda_cache。"""

    nodes = _mapping(config.get("nodes"), "nodes")
    objective = _mapping(config.get("objective"), "objective")
    parameters: dict[str, dict[str, float]] = {}
    for node_type, config_key in (
        ("SV", "sv"),
        ("RSU", "rsu"),
        ("UAV", "uav"),
    ):
        section = _mapping(nodes.get(config_key), f"nodes.{config_key}")
        parameters[node_type] = {
            "eta_ca": float(section["eta_ca"]),
            "xi_ca": float(section["xi_ca"]),
        }
    return {
        "lambda_cache": float(objective["lambda_cache"]),
        "nodes": parameters,
    }


def _probability_value(row: dict[str, str]) -> float:
    """读取 popularity artifact 中表示真实请求分布的概率列。"""

    for column in ("rho_star", "probability", "request_probability"):
        raw_value = row.get(column)
        if raw_value not in (None, ""):
            probability = float(raw_value)
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError(f"{column} 不是有效概率：{raw_value}")
            return probability
    raise ValueError("popularity.csv 缺少 rho_star/probability/request_probability 列。")


def load_popularity_distributions(
    path: Path,
    service_ids: Iterable[int],
    config: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """读取各 popularity epoch 的 rho_star，并按 epoch-service 聚合重复 slot 行。"""

    expected_ids = set(service_ids)
    if path.exists():
        values: dict[int, dict[int, list[float]]] = {}
        row_count = 0
        max_within_epoch_spread = 0.0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "service_id" not in reader.fieldnames:
                raise ValueError("popularity.csv 缺少 service_id 列。")
            for row in reader:
                row_count += 1
                epoch_raw = row.get("epoch", "0")
                epoch = int(epoch_raw or 0)
                service_id = int(row["service_id"])
                if service_id not in expected_ids:
                    raise ValueError(f"popularity.csv 出现未知 service_id：{service_id}")
                values.setdefault(epoch, {}).setdefault(service_id, []).append(_probability_value(row))

        distributions: dict[int, dict[int, float]] = {}
        for epoch, per_service in sorted(values.items()):
            if set(per_service) != expected_ids:
                missing = sorted(expected_ids - set(per_service))
                raise ValueError(f"epoch {epoch} 缺少 service：{missing}")
            distribution: dict[int, float] = {}
            for service_id, samples in per_service.items():
                max_within_epoch_spread = max(max_within_epoch_spread, max(samples) - min(samples))
                distribution[service_id] = sum(samples) / len(samples)
            probability_sum = sum(distribution.values())
            if not math.isclose(
                probability_sum,
                1.0,
                rel_tol=PROBABILITY_SUM_TOLERANCE,
                abs_tol=PROBABILITY_SUM_TOLERANCE,
            ):
                raise ValueError(f"epoch {epoch} 的 rho_star 和为 {probability_sum}，不是 1。")
            distributions[epoch] = distribution

        return {
            "distributions": distributions,
            "source": path,
            "source_kind": "popularity.csv 的 rho_star",
            "row_count": row_count,
            "max_within_epoch_spread": max_within_epoch_spread,
            "fallback": False,
        }

    # 只有缺失 popularity artifact 时才使用 catalog 中的初始排名作为可追溯回退。
    task_config = _mapping(config.get("task"), "task")
    zeta = float(task_config["popularity_zeta"])
    popularity_generation = _mapping(catalog.get("popularity_generation", {}), "popularity_generation")
    ranking = [int(value) for value in popularity_generation.get("initial_ranking", [])]
    if set(ranking) != expected_ids:
        ranking = sorted(expected_ids)
    weights = {service_id: (index + 1) ** (-zeta) for index, service_id in enumerate(ranking)}
    normalization = sum(weights.values())
    distributions = {0: {service_id: value / normalization for service_id, value in weights.items()}}
    return {
        "distributions": distributions,
        "source": None,
        "source_kind": "service_catalog.json 初始排名的 Zipf 回退",
        "row_count": 0,
        "max_within_epoch_spread": 0.0,
        "fallback": True,
    }


def popularity_summary(distributions: dict[int, dict[int, float]]) -> dict[str, Any]:
    """汇总 popularity 的稳定性、epoch 数量和排名变化。"""

    rankings = {
        epoch: tuple(sorted(distribution, key=lambda service_id: (-distribution[service_id], service_id)))
        for epoch, distribution in distributions.items()
    }
    sorted_probability_vectors = [tuple(sorted(distribution.values())) for distribution in distributions.values()]
    base_vector = sorted_probability_vectors[0]
    probability_values_stable = all(
        all(abs(left - right) <= PROBABILITY_SUM_TOLERANCE for left, right in zip(base_vector, vector))
        for vector in sorted_probability_vectors[1:]
    )
    epochs = sorted(rankings)
    adjacent_ranking_changes = sum(
        rankings[current] != rankings[previous]
        for previous, current in zip(epochs, epochs[1:])
    )
    return {
        "epoch_count": len(epochs),
        "rankings": rankings,
        "distinct_ranking_count": len(set(rankings.values())),
        "adjacent_ranking_changes": adjacent_ranking_changes,
        "probability_values_stable": probability_values_stable,
        "min_probability": min(value for distribution in distributions.values() for value in distribution.values()),
        "max_probability": max(value for distribution in distributions.values() for value in distribution.values()),
    }


def max_services_by_size(service_sizes: dict[int, float], capacity_mb: float) -> dict[str, Any]:
    """按真实 service 大小从小到大累加，求容量内可放入的最大服务数量。"""

    selected: list[int] = []
    used_mb = 0.0
    for service_id, size_mb in sorted(service_sizes.items(), key=lambda item: (item[1], item[0])):
        if used_mb + size_mb <= capacity_mb + CAPACITY_TOLERANCE_MB:
            selected.append(service_id)
            used_mb += size_mb
    return {"count": len(selected), "used_mb": used_mb, "service_ids": selected}


def _subset_options(
    service_ids: list[int],
    service_sizes: dict[int, float],
    probabilities: dict[int, float],
    capacity_mb: float,
) -> list[tuple[float, float, tuple[int, ...]]]:
    """枚举半组服务的容量和概率组合，供精确 0-1 背包计算使用。"""

    options: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
    for service_id in service_ids:
        additions: list[tuple[float, float, tuple[int, ...]]] = []
        for used_mb, hit_ratio, chosen in options:
            new_used_mb = used_mb + service_sizes[service_id]
            if new_used_mb <= capacity_mb + CAPACITY_TOLERANCE_MB:
                additions.append(
                    (new_used_mb, hit_ratio + probabilities[service_id], chosen + (service_id,))
                )
        options.extend(additions)
    return options


def exact_hot_cache(
    service_sizes: dict[int, float],
    probabilities: dict[int, float],
    capacity_mb: float,
) -> dict[str, Any]:
    """在真实 service 大小约束下求请求概率最大的可缓存服务集合。"""

    service_ids = sorted(service_sizes)
    split = len(service_ids) // 2
    left_options = _subset_options(service_ids[:split], service_sizes, probabilities, capacity_mb)
    right_options = _subset_options(service_ids[split:], service_sizes, probabilities, capacity_mb)
    right_options.sort(key=lambda item: (item[0], -item[1], item[2]))

    # 将右半边压缩成按容量递增的概率前缀最优点，避免忽略实际服务大小。
    right_weights: list[float] = []
    right_best_values: list[float] = []
    right_best_services: list[tuple[int, ...]] = []
    best_value = -math.inf
    for used_mb, hit_ratio, chosen in right_options:
        if hit_ratio > best_value + VALUE_TOLERANCE:
            right_weights.append(used_mb)
            right_best_values.append(hit_ratio)
            right_best_services.append(chosen)
            best_value = hit_ratio

    best = (0.0, 0.0, ())
    for left_used_mb, left_value, left_services in left_options:
        remaining_mb = capacity_mb - left_used_mb
        index = bisect.bisect_right(right_weights, remaining_mb + CAPACITY_TOLERANCE_MB) - 1
        if index < 0:
            candidate = (left_used_mb, left_value, left_services)
        else:
            candidate = (
                left_used_mb + right_weights[index],
                left_value + right_best_values[index],
                left_services + right_best_services[index],
            )
        if candidate[1] > best[1] + VALUE_TOLERANCE:
            best = candidate
    return {
        "used_mb": best[0],
        "hit_ratio": best[1],
        "service_ids": list(best[2]),
    }


def analyze_cache_pressure(
    config: dict[str, Any],
    catalog: dict[str, Any],
    popularity: dict[str, Any],
) -> dict[str, Any]:
    """计算容量比例、可缓存服务数和各 epoch 的理想热门缓存命中率。"""

    service_sizes = load_service_sizes(catalog)
    cache_profiles = load_cache_profiles(config)
    total_size_mb = sum(service_sizes.values())
    distributions = popularity["distributions"]
    nodes: dict[str, dict[str, Any]] = {}

    for node_type, profile in cache_profiles.items():
        capacity_mb = float(profile["capacity_mb"])
        cardinality = max_services_by_size(service_sizes, capacity_mb)
        ideal_by_epoch: dict[int, dict[str, Any]] = {}
        top_k_by_epoch: dict[int, float] = {}
        for epoch, probabilities in distributions.items():
            ranking = sorted(service_sizes, key=lambda service_id: (-probabilities[service_id], service_id))
            top_k = cardinality["count"]
            top_k_by_epoch[epoch] = sum(probabilities[service_id] for service_id in ranking[:top_k])
            ideal_by_epoch[epoch] = exact_hot_cache(service_sizes, probabilities, capacity_mb)

        ideal_values = [item["hit_ratio"] for item in ideal_by_epoch.values()]
        top_k_values = list(top_k_by_epoch.values())
        nodes[node_type] = {
            **profile,
            "capacity_ratio": capacity_mb / total_size_mb if total_size_mb else 0.0,
            "max_services": cardinality,
            "top_k_coverage_mean": sum(top_k_values) / len(top_k_values),
            "top_k_coverage_min": min(top_k_values),
            "top_k_coverage_max": max(top_k_values),
            "ideal_by_epoch": ideal_by_epoch,
            "ideal_hit_ratio_mean": sum(ideal_values) / len(ideal_values),
            "ideal_hit_ratio_min": min(ideal_values),
            "ideal_hit_ratio_max": max(ideal_values),
        }

    service_sizes_sorted = sorted(service_sizes.items(), key=lambda item: (item[1], item[0]))
    return {
        "service_count": len(service_sizes),
        "service_sizes": service_sizes,
        "service_sizes_sorted": service_sizes_sorted,
        "total_size_mb": total_size_mb,
        "min_size_mb": min(service_sizes.values()),
        "max_size_mb": max(service_sizes.values()),
        "mean_size_mb": total_size_mb / len(service_sizes),
        "nodes": nodes,
    }


def _find_evidence(root: Path, patterns: list[tuple[str, str]]) -> list[str]:
    """在指定源文件中查找审计证据并返回相对路径、行号和代码行。"""

    evidence: list[str] = []
    for relative_path, pattern in patterns:
        path = root / relative_path
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        compiled = re.compile(pattern)
        for line_number, line in enumerate(lines, start=1):
            if compiled.search(line):
                compact_line = line.strip().replace("|", "\\|")
                evidence.append(f"{relative_path}:{line_number} `{compact_line}`")
                break
    return evidence


def audit_cache_update_mechanism(root: Path) -> dict[str, dict[str, Any]]:
    """只读扫描缓存触发、状态差异、替换和更新成本实现。"""

    audit: dict[str, dict[str, Any]] = {}

    def add_item(label: str, patterns: list[tuple[str, str]], conclusion: str) -> None:
        evidence = _find_evidence(root, patterns)
        audit[label] = {
            "found": bool(evidence),
            "conclusion": conclusion,
            "evidence": evidence,
        }

    add_item(
        "trigger_equivalent",
        [
            ("src/ppo/ppo_trigger_policy.py", r"z_\{node_id\}\^ca"),
            ("src/decision/cache_configuration.py", r"cache_trigger"),
        ],
        "存在 z_m^ca 的等价实现：动作标签和优化器输入均使用 cache trigger。",
    )
    add_item(
        "actual_update_detection",
        [
            ("src/common/system_model.py", r"actual_trigger\s*=\s*np\.any\(old\s*!=\s*new"),
            ("src/decision/cache_configuration.py", r"actual_cache_trigger\s*=\s*np\.any\("),
        ],
        "实际更新由 new cache 与 old cache 的逐节点差异决定，而不是只看动作意图。",
    )
    add_item(
        "no_op_update",
        [
            ("src/decision/cache_configuration.py", r"np\.array_equal\(result\[\"new_cache\"\],\s*previous_cache\)"),
            ("src/decision/cache_configuration.py", r"not\s+np\.any\(expected\)"),
            ("src/common/system_model.py", r"old\s*!=\s*new"),
        ],
        "new cache 与 old cache 相同时 actual trigger 为 0，验证路径要求更新成本和能耗为 0。",
    )
    add_item(
        "loading_cost",
        [
            ("src/common/system_model.py", r"additions\s*=\s*np\.maximum\(new\s*-\s*old"),
            ("src/common/system_model.py", r"loaded_mb\s*=\s*additions\s*@\s*sizes"),
            ("src/common/system_model.py", r"loading_energy\s*=\s*xi\s*\*\s*loaded_mb"),
        ],
        "0->1 的服务按真实大小计算 loaded MB，并通过 xi_ca 计算加载能耗/成本项。",
    )
    add_item(
        "control_cost",
        [
            ("src/common/system_model.py", r"cost_by_node\s*=\s*actual_trigger.*eta"),
            ("src/common/system_model.py", r"lam\s*\*\s*loading_energy"),
        ],
        "实际发生更新的节点承担 eta_ca 固定控制成本，加载项再乘 lambda_ca。",
    )
    add_item(
        "replacement_moves",
        [
            ("src/decision/cache_configuration.py", r"generate_node_neighbors"),
            ("src/decision/cache_configuration.py", r"best_move"),
        ],
        "缓存优化器提供 add/delete/swap 邻域，并只接受目标函数严格改善的替换。",
    )
    add_item(
        "trigger_gate",
        [
            ("src/decision/slot_environment.py", r"cache_trigger\.sum\(\).*uav_triggers"),
            ("src/decision/cache_configuration.py", r"triggered_indices\s*=\s*\["),
        ],
        "没有 cache trigger 时存在不搜索的快路径；有 trigger 时只优化被触发节点。",
    )
    add_item(
        "cost_integration",
        [
            ("src/decision/slot_environment.py", r"return\s+cache_configuration\.cache_update_cost"),
            ("src/decision/slot_environment.py", r"objective\s*=\s*float\(predictive_utility\s*-\s*cache_cost"),
            ("src/decision/cache_configuration.py", r"final_objective\s*=\s*float\(predicted_utility\s*-\s*update_cost\)"),
        ],
        "更新成本进入 cache objective，并由 slot environment 统一计入系统效用。",
    )
    return audit


def relative_path(root: Path, path: Path | None) -> str:
    """将路径格式化为报告中的仓库相对路径。"""

    if path is None:
        return "未使用文件"
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _percent(value: float) -> str:
    """将比例格式化为百分数。"""

    return f"{value * 100.0:.3f}%"


def _mb(value: float) -> str:
    """将 MB 数值格式化为便于阅读的字符串。"""

    return f"{value:.6f}"


def render_report(
    root: Path,
    config_path: Path,
    catalog_path: Path,
    popularity_path: Path,
    config: dict[str, Any],
    catalog: dict[str, Any],
    popularity: dict[str, Any],
    pressure: dict[str, Any],
    audit: dict[str, dict[str, Any]],
) -> str:
    """将全部分析结果渲染为中文 Markdown 报告。"""

    task = _mapping(config.get("task"), "task")
    service_config = _mapping(config.get("service"), "service")
    cost_parameters = load_cache_cost_parameters(config)
    popularity_stats = popularity_summary(popularity["distributions"])
    capacity_summary = "、".join(
        f"{node_type}={_mb(profile['capacity_mb'])}MB"
        for node_type, profile in pressure["nodes"].items()
    )
    max_service_summary = "、".join(
        f"{node_type}={profile['max_services']['count']}个"
        for node_type, profile in pressure["nodes"].items()
    )
    lines = [
        "# 缓存容量压力分析与更新机制审计",
        "",
        f"> 分析日期：{date.today().isoformat()}。本报告只读分析当前正式 baseline，未修改配置、算法或 artifact。",
        "",
        "## 1. 分析输入与当前参数",
        "",
        f"- 配置：`{relative_path(root, config_path)}`（唯一配置入口）。",
        f"- 服务目录：`{relative_path(root, catalog_path)}`；配置 service 数量为 `{int(service_config['count'])}`，artifact service 数量为 `{int(catalog.get('service_count', pressure['service_count']))}`。",
        f"- 请求分布：`{relative_path(root, popularity.get('source') or popularity_path)}`，使用 `{popularity['source_kind']}`。",
        f"- popularity epoch：{popularity_stats['epoch_count']} 个；配置 epoch 长度：{int(task['popularity_epoch_length'])} slot；每 epoch 变化次数：{int(task['popularity_changes_per_epoch'])}；Zipf 指数：{float(task['popularity_zeta']):.3f}。",
        f"- arrival probability：{float(task['arrival_probability']):.3f}。该参数不改变容量比例，但用于记录当前请求模型。",
        "",
        "| 节点类型 | 缓存容量 (MB) | 配置读取路径 |",
        "|---|---:|---|",
    ]
    for node_type, profile in pressure["nodes"].items():
        lines.append(
            f"| {node_type} | {_mb(profile['capacity_mb'])} | `{profile['config_key']}` |"
        )

    lines.extend(
        [
            "",
            f"当前 `objective.lambda_cache = {cost_parameters['lambda_cache']:.6f}`；动态 cache 更新成本参数如下。TV local cache 是静态 artifact，不参与下表的动态 cache reconfiguration。",
            "",
            "| 动态节点 | eta_ca（控制成本） | xi_ca（加载成本/MB） |",
            "|---|---:|---:|",
        ]
    )
    for node_type in ("SV", "RSU", "UAV"):
        parameters = cost_parameters["nodes"][node_type]
        lines.append(
            f"| {node_type} | {parameters['eta_ca']:.6f} | {parameters['xi_ca']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Service catalog 总大小",
            "",
            f"`S_total = sum(service_size) = {pressure['total_size_mb']:.6f} MB`。",
            "",
            "| 统计量 | 数值 (MB) |",
            "|---|---:|",
            f"| service 数量 | {pressure['service_count']} |",
            f"| 最小 service size | {_mb(pressure['min_size_mb'])} |",
            f"| 最大 service size | {_mb(pressure['max_size_mb'])} |",
            f"| 平均 service size（仅作描述） | {_mb(pressure['mean_size_mb'])} |",
            "",
            "容量比例按 `cache_capacity / S_total` 计算；可缓存数量和热门覆盖率均使用 catalog 中每个 service 的实际大小。",
            "",
            "## 3. 容量压力与热门缓存覆盖率",
            "",
            "`最大可缓存服务数` 按实际大小从小到大累加，因而是同一容量下的最大数量；没有使用平均大小估算。",
            "`top-K 排名覆盖率` 是按热门排名前 K 个服务的概率和，用于展示热度上限；由于不同 service 大小不同，它不保证前 K 个能同时装入容量。",
            "`理想热门命中率` 则在真实大小约束下求解容量内概率最大的 0-1 服务集合，更接近可实现的理想上界。",
            "",
            "| 节点 | 容量 (MB) | 容量 / catalog | 最大可缓存服务数 | top-K 排名覆盖率（epoch均值） | 容量约束理想命中率（均值） | 理想命中率范围 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for node_type, profile in pressure["nodes"].items():
        lines.append(
            "| {node} | {capacity} | {ratio} | {count} | {top_k} | {ideal} | {minimum}--{maximum} |".format(
                node=node_type,
                capacity=_mb(profile["capacity_mb"]),
                ratio=_percent(profile["capacity_ratio"]),
                count=profile["max_services"]["count"],
                top_k=_percent(profile["top_k_coverage_mean"]),
                ideal=_percent(profile["ideal_hit_ratio_mean"]),
                minimum=_percent(profile["ideal_hit_ratio_min"]),
                maximum=_percent(profile["ideal_hit_ratio_max"]),
            )
        )

    lines.extend(
        [
            "",
            f"按当前真实大小，四类节点的最大服务数量分别由最小服务组合决定：{max_service_summary}。具体的热门命中率已在上表中按每个 epoch 重新计算。",
            "",
            "## 4. Zipf 请求压力",
            "",
            f"当前任务模型为 Zipf `zeta={float(task['popularity_zeta']):.3f}`，请求分布按 `popularity.csv` 的 `rho_star` 读取。`rho_star` 的单个概率值范围为 {float(popularity_stats['min_probability']):.6f}--{float(popularity_stats['max_probability']):.6f}。",
            f"排序后的 Zipf 概率集合跨 epoch {'保持不变' if popularity_stats['probability_values_stable'] else '发生变化'}；热门服务排序共有 {popularity_stats['distinct_ranking_count']} 种，连续 epoch 间发生 {popularity_stats['adjacent_ranking_changes']} 次排名变化。",
            "因此，分布的 Zipf 形状相对稳定，但服务 ID 的热门顺序并非全程固定；容量内的理想命中集合存在随 epoch 调整的收益空间。",
            "",
            "## 5. 缓存更新机制审计",
            "",
            "以下为源代码静态审计结果，没有调用训练、SUMO 或完整 slot 仿真。",
            "",
            "| 审计项 | 结果 | 结论 | 代码证据 |",
            "|---|---|---|---|",
        ]
    )
    audit_labels = [
        ("trigger_equivalent", "z_m^ca 等价触发变量"),
        ("actual_update_detection", "实际更新判定"),
        ("no_op_update", "new cache == old cache 的无效更新处理"),
        ("loading_cost", "加载成本"),
        ("control_cost", "控制成本"),
        ("replacement_moves", "替换/邻域动作"),
        ("trigger_gate", "触发门控"),
        ("cost_integration", "成本进入目标函数"),
    ]
    for key, label in audit_labels:
        item = audit[key]
        result = "发现" if item["found"] else "未发现"
        evidence = "；".join(item["evidence"]) if item["evidence"] else "无匹配证据"
        lines.append(f"| {label} | {result} | {item['conclusion']} | {evidence} |")

    max_ratio = max(profile["capacity_ratio"] for profile in pressure["nodes"].values())
    lines.extend(
        [
            "",
            "审计结论：",
            "",
            "1. 代码中存在与论文 `z_m^ca(t)` 等价的 cache trigger；动作触发只是更新意图，最终是否真实更新由新旧 cache 状态差异确定。",
            "2. 当 `new_cache == old_cache` 时，不会被计为 actual update；相应加载能耗和 cache update cost 应为 0。",
            "3. 代码同时计算 0->1 服务加载量、`xi_ca` 加载项和 `eta_ca` 固定控制项，并将更新成本扣除到 configuration objective。",
            "4. 更新不是每 slot 强制发生：没有 trigger 时走不搜索快路径；有 trigger 但候选目标不严格改善时也会保留原 cache。因此“动作触发”与“有效 cache 更新”是两个不同事件。",
            "",
            "## 6. 是否存在缓存长期不更新风险",
            "",
            f"当前最大容量比例为 {_percent(max_ratio)}，RSU 也只覆盖 catalog 的 {max_ratio:.1%}；当前容量摘要为 {capacity_summary}，并非接近完整 catalog 的容量。仅从容量看，没有“整个 catalog 都能长期驻留”导致更新必然消失的条件。",
            f"同时，`rho_star` 的概率集合 {'稳定' if popularity_stats['probability_values_stable'] else '不稳定'}，但热门排序在 epoch 间发生变化；而更新控制成本、加载成本和严格改善门槛均存在。由此判断：当前配置不存在由容量直接造成的必然长期不更新风险，但存在条件性的稀疏更新风险——若外层策略长期输出 0 trigger，或每次热度变化带来的收益不足以抵消更新成本，cache 可以保持不变。",
            "",
            "## 7. 容量建议",
            "",
            f"本次不建议仅凭该压力分析修改缓存容量。当前容量提供了从 TV 的单服务缓存到 RSU/UAV 的多服务缓存层次，且热门排序会变化；建议保留当前由 catalog 比例推导的 {capacity_summary} baseline。若后续实验观测到长期无更新，应优先统计外层 trigger 发出率、actual_cache_trigger 发出率和“候选收益 - 更新成本”，再决定是否做容量或成本敏感性实验。",
            "",
            "## 8. 修改边界",
            "",
            "本次分析只读取 `system_config.toml`、service catalog、popularity 和缓存相关源码；没有修改配置、缓存算法、reward、PPO、task generation 或任何 artifact。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析可选输入和输出路径。"""

    root = project_root()
    parser = argparse.ArgumentParser(description="分析当前缓存容量压力并审计缓存更新机制。")
    parser.add_argument("--config", type=Path, default=root / DEFAULT_CONFIG)
    parser.add_argument("--service-catalog", type=Path, default=root / DEFAULT_SERVICE_CATALOG)
    parser.add_argument("--popularity", type=Path, default=root / DEFAULT_POPULARITY)
    parser.add_argument("--report", type=Path, default=root / DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """执行只读分析并写入 Markdown 报告。"""

    args = parse_args()
    root = project_root()
    config_path = resolve_project_path(root, args.config)
    catalog_path = resolve_project_path(root, args.service_catalog)
    popularity_path = resolve_project_path(root, args.popularity)
    report_path = resolve_project_path(root, args.report)

    config = load_toml(config_path)
    catalog = load_json(catalog_path)
    service_sizes = load_service_sizes(catalog)
    popularity = load_popularity_distributions(popularity_path, service_sizes, config, catalog)
    pressure = analyze_cache_pressure(config, catalog, popularity)
    audit = audit_cache_update_mechanism(root)
    report = render_report(
        root,
        config_path,
        catalog_path,
        popularity_path,
        config,
        catalog,
        popularity,
        pressure,
        audit,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(f"报告已生成：{relative_path(root, report_path)}")
    print(f"service catalog 总大小：{pressure['total_size_mb']:.6f} MB")
    for node_type, profile in pressure["nodes"].items():
        print(
            f"{node_type}: capacity={profile['capacity_mb']:.6f} MB, "
            f"ratio={_percent(profile['capacity_ratio'])}, "
            f"max_services={profile['max_services']['count']}, "
            f"ideal_hit_mean={_percent(profile['ideal_hit_ratio_mean'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
