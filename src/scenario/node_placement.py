#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据 SUMO 路网拓扑自动生成 RSU/UAV 初始部署位置。

节点候选位置直接来自 SUMO net.xml 中的 junction，不在代码或 TOML 中维护
人工坐标。运行本脚本即可按照当前 ``nodes.rsu.count`` 和 ``nodes.uav.count``
重新生成 ``paths.node_positions`` 指向的统一位置文件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    # 兼容未内置 tomllib 的 Python 环境。
    import tomli as tomllib  # type: ignore[no-redef]


def find_project_root() -> Path:
    """从脚本位置或当前工作目录定位项目根目录。"""

    candidates = (
        Path(__file__).resolve().parents[2],
        Path.cwd().resolve(),
    )
    for candidate in candidates:
        if (candidate / "system_config.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "无法定位项目根目录：期望找到 system_config.toml 和 data/。"
    )


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """将配置中的相对路径解析为项目内的绝对路径。"""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _resolve_config_path(project_root: Path, value: str | Path) -> Path:
    """解析配置文件路径，并兼容从项目外启动脚本的情况。"""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    working_candidate = (Path.cwd() / candidate).resolve(strict=False)
    if working_candidate.is_file():
        return working_candidate
    return resolve_path(project_root, candidate)


def _get_config_value(config: dict[str, Any], dotted_key: str) -> Any:
    """读取点号分隔的配置项，并在配置缺失时给出明确错误。"""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"配置缺少 {dotted_key}。")
        current = current[part]
    return current


def load_raw_config(config_path: Path) -> dict[str, Any]:
    """读取原始 TOML 配置，避免位置生成前触发其他实验输入校验。"""

    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if not isinstance(config, dict):
        raise ValueError("system_config.toml 根对象必须是 table。")
    return config


def _load_sumolib() -> Any:
    """加载 SUMO 官方 sumolib，并兼容仅设置 SUMO_HOME 的安装方式。"""

    try:
        import sumolib

        return sumolib
    except ModuleNotFoundError as import_error:
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise ModuleNotFoundError(
                "无法导入 sumolib；请安装 SUMO 或设置 SUMO_HOME。"
            ) from import_error

        sumo_root = Path(sumo_home).expanduser()
        tools_dir = sumo_root
        if not (tools_dir / "sumolib").is_dir():
            tools_dir = sumo_root / "tools"
        if not (tools_dir / "sumolib").is_dir():
            raise ModuleNotFoundError(
                f"SUMO_HOME 下未找到 sumolib：{sumo_root}"
            ) from import_error
        sys.path.insert(0, str(tools_dir))
        try:
            import sumolib
        except ModuleNotFoundError as sumolib_error:
            raise ModuleNotFoundError(
                f"无法从 SUMO tools 目录导入 sumolib：{tools_dir}"
            ) from sumolib_error
        return sumolib


def _normalise_boundary(raw_boundary: Any) -> dict[str, float]:
    """将 sumolib 返回的边界统一为 min/max 字段。"""

    try:
        if (
            isinstance(raw_boundary, (tuple, list))
            and len(raw_boundary) == 2
            and all(
                isinstance(point, (tuple, list)) and len(point) >= 2
                for point in raw_boundary
            )
        ):
            (min_x, min_y), (max_x, max_y) = raw_boundary
        elif isinstance(raw_boundary, (tuple, list)) and len(raw_boundary) == 4:
            min_x, min_y, max_x, max_y = raw_boundary
        else:
            raise ValueError
        boundary = {
            "min_x": float(min_x),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "max_y": float(max_y),
        }
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"SUMO net boundary 格式错误：{raw_boundary}") from exc

    if not all(math.isfinite(value) for value in boundary.values()):
        raise ValueError(f"SUMO net boundary 不能包含 NaN/Inf：{raw_boundary}")
    if boundary["min_x"] >= boundary["max_x"] or boundary["min_y"] >= boundary["max_y"]:
        raise ValueError(f"SUMO net boundary 非法：{raw_boundary}")
    return boundary


def read_sumo_network(net_file: Path) -> tuple[Any, dict[str, float]]:
    """使用 sumolib 读取 SUMO 路网对象和实际空间边界。"""

    if not net_file.is_file():
        raise FileNotFoundError(f"SUMO net 文件不存在：{net_file}")
    sumolib = _load_sumolib()
    # 读取完整路网对象，后续从 junction 的入边和出边计算道路连接度。
    network = sumolib.net.readNet(str(net_file))
    boundary = _normalise_boundary(network.getBoundary())
    return network, boundary


def score_junction(junction: Any) -> int:
    """按与 junction 相连的非内部道路数量计算节点评分。"""

    connected_edge_ids: set[str] = set()
    # 入边和出边可能指向同一条道路，使用 ID 去重得到道路连接度。
    for edge in (*junction.getIncoming(), *junction.getOutgoing()):
        get_function = getattr(edge, "getFunction", None)
        if callable(get_function) and get_function() == "internal":
            continue
        connected_edge_ids.add(str(edge.getID()))
    return len(connected_edge_ids)


def extract_junction_candidates(network: Any) -> list[dict[str, Any]]:
    """读取 SUMO 路网中的 junction，构造 RSU 候选点列表。"""

    candidates: list[dict[str, Any]] = []
    # 读取 SUMO 路网中的所有交叉节点，作为 RSU 候选部署位置。
    for junction in network.getNodes():
        junction_id = str(junction.getID())
        coordinates = junction.getCoord()
        if not isinstance(coordinates, (tuple, list)) or len(coordinates) < 2:
            continue
        x, y = float(coordinates[0]), float(coordinates[1])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        candidates.append(
            {
                "id": junction_id,
                "x": x,
                "y": y,
                "score": score_junction(junction),
            }
        )
    candidates.sort(key=lambda item: str(item["id"]))
    if not candidates:
        raise ValueError("SUMO 路网中没有可用于 RSU 部署的 junction。")
    return candidates


def _validate_count(name: str, value: int) -> int:
    """校验节点数量为正整数。"""

    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} 必须是正整数：{value}")
    return int(value)


def _validate_positive_number(name: str, value: float) -> float:
    """校验部署参数为有限正数。"""

    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} 必须是有限正数：{value}")
    return number


def _distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    """计算两个路网候选点的二维欧氏距离。"""

    return math.hypot(
        float(first["x"]) - float(second["x"]),
        float(first["y"]) - float(second["y"]),
    )


def select_rsu_candidates(
    candidates: list[dict[str, Any]],
    num_rsu: int,
) -> list[dict[str, Any]]:
    """按最高连接度起点和最大最小距离策略选择 RSU 候选点。"""

    count = _validate_count("num_rsu", num_rsu)
    if count > len(candidates):
        raise ValueError(
            f"num_rsu={count} 超过可用 junction 候选点数量 {len(candidates)}。"
        )

    # 第一个 RSU 选择道路连接度最高的节点，ID 用于稳定处理同分情况。
    first = sorted(
        candidates,
        key=lambda item: (-float(item["score"]), str(item["id"])),
    )[0]
    selected = [first]
    remaining = [candidate for candidate in candidates if candidate is not first]

    while len(selected) < count:
        # 最大化候选点到已选集合的最小距离，避免 RSU 连续集中在相邻节点。
        def selection_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
            """为候选点生成距离优先、评分次优先的稳定排序键。"""

            nearest_distance = min(_distance(candidate, chosen) for chosen in selected)
            return (-nearest_distance, -float(candidate["score"]), str(candidate["id"]))

        next_candidate = sorted(remaining, key=selection_key)[0]
        selected.append(next_candidate)
        remaining.remove(next_candidate)
    return selected


def _rounded(value: float) -> float:
    """将坐标保留到足够的十进制位，避免 JSON 出现无意义浮点噪声。"""

    return float(f"{float(value):.9f}")


def _inside_xy(boundary: dict[str, float], x: float, y: float) -> bool:
    """判断二维坐标是否位于 SUMO 路网边界内。"""

    return (
        boundary["min_x"] <= x <= boundary["max_x"]
        and boundary["min_y"] <= y <= boundary["max_y"]
    )


def _uav_y_with_offset(
    base_y: float,
    boundary: dict[str, float],
    offset: float,
) -> float:
    """优先向边界上方偏移 UAV，越界时选择仍在路网区域内的反向偏移。"""

    upward = base_y + offset
    if boundary["min_y"] <= upward <= boundary["max_y"]:
        return upward
    downward = base_y - offset
    if boundary["min_y"] <= downward <= boundary["max_y"]:
        return downward
    # 当配置的偏移大于可用垂直空间时，选择离基准点更远的合法边界。
    distance_to_min = base_y - boundary["min_y"]
    distance_to_max = boundary["max_y"] - base_y
    return boundary["min_y"] if distance_to_min >= distance_to_max else boundary["max_y"]


def _interpolate_rsu_reference(
    rsu_candidates: list[dict[str, Any]],
    fraction: float,
) -> tuple[float, float, str]:
    """沿按空间排序的 RSU 折线插值，返回位置和最近的 RSU 标识。"""

    if len(rsu_candidates) == 1:
        candidate = rsu_candidates[0]
        return float(candidate["x"]), float(candidate["y"]), str(candidate["id"])

    scaled = fraction * (len(rsu_candidates) - 1)
    left_index = min(int(math.floor(scaled)), len(rsu_candidates) - 1)
    right_index = min(left_index + 1, len(rsu_candidates) - 1)
    local_fraction = scaled - left_index
    left = rsu_candidates[left_index]
    right = rsu_candidates[right_index]
    x = float(left["x"]) + local_fraction * (float(right["x"]) - float(left["x"]))
    y = float(left["y"]) + local_fraction * (float(right["y"]) - float(left["y"]))
    # 以线段中点确定插值位置对应的初始 RSU，保证元数据稳定且可解释。
    anchor = left if local_fraction <= 0.5 else right
    return x, y, str(anchor["id"])


def _single_rsu_uav_x(
    rsu_x: float,
    index: int,
    num_uav: int,
    boundary: dict[str, float],
) -> float:
    """在只有一个 RSU 时沿路网横向展开 UAV，避免退化为重合位置。"""

    width = boundary["max_x"] - boundary["min_x"]
    # 单个 RSU 没有可供插值的线段，使用以 RSU 为中心的横向展开作为退化处理。
    spacing = width / (num_uav + 1)
    center_index = (num_uav - 1) / 2.0
    proposed = rsu_x + (index - center_index) * spacing
    if boundary["min_x"] <= proposed <= boundary["max_x"]:
        return proposed
    # RSU 位于边界附近时，采用同一边界范围内的均匀采样保证所有点合法。
    return boundary["min_x"] + (index + 1) / (num_uav + 1) * width


def generate_uav_positions(
    selected_rsus: list[dict[str, Any]],
    num_uav: int,
    boundary: dict[str, float],
    offset_fraction: float,
    uav_altitude: float,
) -> dict[str, dict[str, Any]]:
    """根据已选 RSU 位置生成 UAV 初始位置和对应 RSU 元数据。"""

    count = _validate_count("num_uav", num_uav)
    offset_ratio = _validate_positive_number("nodes.uav.deployment_offset_fraction", offset_fraction)
    altitude = _validate_positive_number("nodes.uav.deployment_altitude_m", uav_altitude)
    width = boundary["max_x"] - boundary["min_x"]
    height = boundary["max_y"] - boundary["min_y"]
    offset = offset_ratio * max(width, height)

    ordered_rsus = sorted(
        selected_rsus,
        key=lambda item: (float(item["x"]), float(item["y"]), str(item["id"])),
    )
    # 位置插值按空间顺序进行，但元数据仍需指向输出文件中的 RSU 编号。
    rsu_output_ids = {
        str(candidate["id"]): f"RSU_{index}"
        for index, candidate in enumerate(selected_rsus)
    }
    generated: dict[str, dict[str, Any]] = {}
    for index in range(count):
        if count == len(ordered_rsus):
            # 数量相等时保持 RSU 与 UAV 的一一对应关系。
            base = selected_rsus[index]
            base_x, base_y, initial_rsu = (
                float(base["x"]),
                float(base["y"]),
                rsu_output_ids[str(base["id"])],
            )
        else:
            # UAV 数量不等于 RSU 数量时，在所有 RSU 之间取均匀插值位置。
            fraction = 0.5 if count == 1 else index / (count - 1)
            base_x, base_y, initial_rsu = _interpolate_rsu_reference(
                ordered_rsus,
                fraction,
            )
            initial_rsu = rsu_output_ids[initial_rsu]
            if len(ordered_rsus) == 1 and count > 1:
                base_x = _single_rsu_uav_x(base_x, index, count, boundary)

        uav_x = base_x
        uav_y = _uav_y_with_offset(base_y, boundary, offset)
        if not _inside_xy(boundary, uav_x, uav_y):
            raise AssertionError(f"UAV_{index} 坐标超出 SUMO 路网区域：{uav_x}, {uav_y}")
        generated[f"UAV_{index}"] = {
            "position": [_rounded(uav_x), _rounded(uav_y), _rounded(altitude)],
            "initial_rsu": initial_rsu,
        }
    return generated


def build_node_positions_payload(
    net_file: Path,
    project_root: Path,
    boundary: dict[str, float],
    selected_rsus: list[dict[str, Any]],
    uav_positions: dict[str, dict[str, Any]],
    rsu_altitude: float,
) -> dict[str, Any]:
    """组装统一节点位置文件，保留边界字段以兼容现有读取接口。"""

    altitude = _validate_positive_number("nodes.rsu.deployment_altitude_m", rsu_altitude)
    try:
        net_file_value = net_file.relative_to(project_root).as_posix()
    except ValueError:
        net_file_value = net_file.as_posix()

    rsu_positions: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(selected_rsus):
        node_id = f"RSU_{index}"
        x, y = float(candidate["x"]), float(candidate["y"])
        if not _inside_xy(boundary, x, y):
            raise AssertionError(f"{node_id} 坐标超出 SUMO 路网区域：{x}, {y}")
        rsu_positions[node_id] = {
            "position": [_rounded(x), _rounded(y), _rounded(altitude)],
            "source_junction": str(candidate["id"]),
        }

    return {
        "network": {
            "net_file": net_file_value,
        },
        # 保留顶层边界，兼容现有 common.project_config 的区域校验接口。
        "network_boundary": {
            key: _rounded(value) for key, value in boundary.items()
        },
        "rsu": rsu_positions,
        "uav": uav_positions,
    }


def write_node_positions(output_file: Path, payload: dict[str, Any]) -> None:
    """写入统一节点位置 JSON 文件。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """构造节点部署脚本的命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="根据 SUMO 路网 junction 拓扑生成 RSU/UAV 节点位置。"
    )
    parser.add_argument(
        "--config",
        default="system_config.toml",
        help="统一 TOML 配置路径，默认使用项目根目录下的 system_config.toml。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="节点位置输出路径，默认读取 paths.node_positions。",
    )
    parser.add_argument(
        "--num-rsu",
        type=int,
        default=None,
        help="可选的 RSU 数量覆盖；不提供时读取 nodes.rsu.count。",
    )
    parser.add_argument(
        "--num-uav",
        type=int,
        default=None,
        help="可选的 UAV 数量覆盖；不提供时读取 nodes.uav.count。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取统一配置、生成节点位置并输出摘要。"""

    args = build_argument_parser().parse_args(argv)
    project_root = find_project_root()
    config_path = _resolve_config_path(project_root, args.config)
    config = load_raw_config(config_path)

    net_file = resolve_path(project_root, _get_config_value(config, "paths.sumo_net"))
    output_value = args.output or _get_config_value(config, "paths.node_positions")
    output_file = resolve_path(project_root, output_value)
    num_rsu = (
        args.num_rsu
        if args.num_rsu is not None
        else int(_get_config_value(config, "nodes.rsu.count"))
    )
    num_uav = (
        args.num_uav
        if args.num_uav is not None
        else int(_get_config_value(config, "nodes.uav.count"))
    )
    rsu_altitude = float(_get_config_value(config, "nodes.rsu.deployment_altitude_m"))
    uav_altitude = float(_get_config_value(config, "nodes.uav.deployment_altitude_m"))
    offset_fraction = float(
        _get_config_value(config, "nodes.uav.deployment_offset_fraction")
    )

    network, boundary = read_sumo_network(net_file)
    candidates = extract_junction_candidates(network)
    selected_rsus = select_rsu_candidates(candidates, num_rsu)
    uav_positions = generate_uav_positions(
        selected_rsus,
        num_uav,
        boundary,
        offset_fraction,
        uav_altitude,
    )
    payload = build_node_positions_payload(
        net_file,
        project_root,
        boundary,
        selected_rsus,
        uav_positions,
        rsu_altitude,
    )
    write_node_positions(output_file, payload)

    print(f"SUMO network: {net_file}")
    print(f"junction candidates: {len(candidates)}")
    print(f"node positions: {output_file}")
    print(f"network boundary: {payload['network_boundary']}")
    print(f"RSU: {payload['rsu']}")
    print(f"UAV: {payload['uav']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
