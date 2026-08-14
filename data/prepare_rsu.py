# -*- coding: utf-8 -*-
"""
从现有 SUMO 地图中生成固定 RSU 部署文件。

规则：
    提取 SUMO net 中 type 以 "traffic_light" 开头的非内部 junction，
    作为固定 RSU 位置。

输出：
    data/rsu/rsu_nodes.csv

若地图中不存在 traffic_light junction，则明确报错，
不会自动改用其他路口类型。
"""

from __future__ import annotations

import csv
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NET_FILE = ROOT / "data" / "sumo" / "map" / "osm.net.xml.gz"
OUTPUT_FILE = ROOT / "data" / "rsu" / "rsu_nodes.csv"


def open_net_file(path: Path):
    """同时支持 .xml 和 .xml.gz。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def extract_signalized_junctions(net_file: Path) -> list[dict]:
    """提取固定信号交叉口。"""
    if not net_file.is_file():
        raise FileNotFoundError(f"SUMO 路网不存在：{net_file}")

    rows = []

    with open_net_file(net_file) as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag

            if "}" in tag:
                tag = tag.rsplit("}", 1)[1]

            if tag != "junction":
                elem.clear()
                continue

            junction_id = elem.attrib.get("id", "")
            junction_type = elem.attrib.get("type", "")

            if (
                junction_id.startswith(":")
                or not junction_type.startswith("traffic_light")
            ):
                elem.clear()
                continue

            x_text = elem.attrib.get("x")
            y_text = elem.attrib.get("y")

            if x_text is None or y_text is None:
                elem.clear()
                continue

            rows.append(
                {
                    "node_id": f"rsu_{junction_id}",
                    "junction_id": junction_id,
                    "junction_type": junction_type,
                    "x": float(x_text),
                    "y": float(y_text),
                }
            )

            elem.clear()

    rows.sort(key=lambda row: row["node_id"])
    return rows


def main():
    rows = extract_signalized_junctions(NET_FILE)

    if not rows:
        raise RuntimeError(
            "当前 SUMO 地图中没有找到 traffic_light junction。"
            "本脚本不会自动改用其他路口类型。"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fields = ["node_id", "junction_id", "junction_type", "x", "y"]

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "x": f"{row['x']:.6f}",
                    "y": f"{row['y']:.6f}",
                }
            )

    print("=" * 72)
    print("RSU 固定部署生成完成")
    print("=" * 72)
    print(f"SUMO 路网      : {NET_FILE}")
    print(f"RSU 数量       : {len(rows)}")
    print(f"输出文件       : {OUTPUT_FILE}")
    print()
    print("RSU 固定位置：")

    for row in rows:
        print(
            f"  {row['node_id']}: "
            f"({row['x']:.3f}, {row['y']:.3f}), "
            f"type={row['junction_type']}"
        )


if __name__ == "__main__":
    main()
