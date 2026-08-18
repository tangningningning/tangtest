# -*- coding: utf-8 -*-
"""
固定 RSU 状态。

数据源：
    data/rsu/rsu_deployment.csv

当前职责：
1. 读取当前正式接入的固定 RSU；
2. 转换为 p3_solver 的统一边缘节点格式；
3. RSU 只提供固定二维位置，不参与 P2。

RSU 的缓存状态由 cache_model / P1 管理；
RSU 的任务执行由 P3 管理。
"""

from __future__ import annotations

import csv
import config


def load_rsu_nodes() -> list[dict]:
    """读取固定 RSU 节点。"""
    path = config.RSU_DEPLOYMENT_FILE

    if not path.is_file():
        raise FileNotFoundError(f"RSU 部署文件不存在：{path}")

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

    expected = set(config.RSU_SELECTED_NODE_IDS)
    actual = {node["node_id"] for node in nodes}

    if actual != expected:
        raise RuntimeError(
            "RSU deployment 与 config 中的 RSU_SELECTED_NODE_IDS 不一致。"
        )

    return sorted(nodes, key=lambda node: node["node_id"])
