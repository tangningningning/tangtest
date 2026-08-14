# -*- coding: utf-8 -*-
"""
固定 RSU 状态读取模块。
"""

from __future__ import annotations

import csv
import config


def load_rsu_nodes() -> list[dict]:
    """读取固定 RSU，并转换成 P3 统一节点格式。"""
    path = config.RSU_DEPLOYMENT_FILE

    if not path.is_file():
        raise FileNotFoundError(
            f"RSU 部署文件不存在：{path}\n"
            "请先运行：python data\\prepare_rsu.py"
        )

    nodes = []
    seen = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            node_id = str(row["node_id"])

            if node_id in seen:
                raise RuntimeError(f"RSU node_id 重复：{node_id}")

            seen.add(node_id)

            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": "RSU",
                    "cpu_hz": config.RSU_CPU_HZ,
                    "position": (
                        float(row["x"]),
                        float(row["y"]),
                    ),
                }
            )

    if not nodes:
        raise RuntimeError("RSU 部署文件为空。")

    return nodes
