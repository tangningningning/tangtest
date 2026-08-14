#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 SUMO 轨迹中适合固定 TV/SV 角色的 episode 时间窗口。

放置位置：
    adaptive_cache_mec/data/analyze_episode_windows.py

默认输入：
    data/sumo/processed/vehicle_trajectory_stats.csv

默认输出：
    data/sumo/processed/episode_window_capacity.csv

如果指定：
    --episode-slots T

还会输出：
    data/sumo/processed/eligible_vehicles_T{T}.csv

定义：
对于 episode 窗口 [start_slot, end_slot]，
只有满足

    first_slot <= start_slot
    last_slot  >= end_slot

的车辆，才算完整覆盖该 episode，可以进入固定 TV/SV 候选池。

注意：
- 这里是离线场景构造，不是在线算法使用未来信息。
- 后续所有算法都应使用同一个固定 episode 和同一候选车辆池。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def project_root_from_script() -> Path:
    script = Path(__file__).resolve()

    if script.parent.name == "data":
        root = script.parent.parent
        if (root / "data" / "sumo").is_dir():
            return root

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "sumo").is_dir():
        return cwd

    raise FileNotFoundError(
        "无法定位项目根目录。请从项目根目录运行，"
        "或将脚本放在项目 data/ 目录。"
    )


def load_stats(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "vehicle_id",
            "first_slot",
            "last_slot",
            "num_records",
            "missing_slots",
            "continuous",
        }

        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"统计文件缺少字段：{sorted(missing)}"
            )

        for row in reader:
            rows.append(
                {
                    "vehicle_id": str(row["vehicle_id"]),
                    "first_slot": int(row["first_slot"]),
                    "last_slot": int(row["last_slot"]),
                    "num_records": int(row["num_records"]),
                    "missing_slots": int(row["missing_slots"]),
                    "continuous": str(row["continuous"]).strip().lower()
                    in {"true", "1", "yes"},
                }
            )

    if not rows:
        raise ValueError(f"统计文件为空：{path}")

    # 固定角色候选必须自身连续。
    bad = [
        r["vehicle_id"]
        for r in rows
        if (not r["continuous"]) or r["missing_slots"] != 0
    ]
    if bad:
        raise ValueError(
            "发现存在内部缺失 slot 的车辆，当前脚本不将其纳入固定角色分析："
            + ", ".join(bad[:20])
        )

    return rows


def eligible_for_window(
    vehicles: list[dict],
    start_slot: int,
    end_slot: int,
) -> list[dict]:
    return [
        v
        for v in vehicles
        if v["first_slot"] <= start_slot
        and v["last_slot"] >= end_slot
    ]


def best_window_for_length(
    vehicles: list[dict],
    episode_slots: int,
    global_first: int,
    global_last: int,
) -> tuple[int, int, list[dict]]:
    """
    找到给定 episode_slots 下，完整覆盖车辆数最多的窗口。
    并列时选择最早的 start_slot，保证结果确定。
    """
    if episode_slots <= 0:
        raise ValueError("episode_slots 必须大于 0。")

    max_possible = global_last - global_first + 1
    if episode_slots > max_possible:
        raise ValueError(
            f"episode_slots={episode_slots} 超过全局轨迹长度 {max_possible}。"
        )

    best_start = global_first
    best_end = global_first + episode_slots - 1
    best_eligible: list[dict] = []

    latest_start = global_last - episode_slots + 1

    for start in range(global_first, latest_start + 1):
        end = start + episode_slots - 1
        eligible = eligible_for_window(vehicles, start, end)

        if len(eligible) > len(best_eligible):
            best_start = start
            best_end = end
            best_eligible = eligible

    return best_start, best_end, best_eligible


def numeric_id_key(vehicle_id: str):
    try:
        return (0, int(vehicle_id))
    except ValueError:
        return (1, vehicle_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分析不同 episode 长度下可完整覆盖窗口的车辆数量。"
    )
    parser.add_argument(
        "--episode-slots",
        type=int,
        default=None,
        help=(
            "可选。指定 episode 长度 T（slot 数）。"
            "指定后会导出该 T 下最佳窗口的候选车辆。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )
    args = parser.parse_args()

    root = project_root_from_script()
    processed_dir = root / "data" / "sumo" / "processed"

    stats_path = processed_dir / "vehicle_trajectory_stats.csv"
    capacity_path = processed_dir / "episode_window_capacity.csv"

    if not stats_path.is_file():
        raise FileNotFoundError(
            f"没有找到轨迹统计文件：{stats_path}\n"
            "请先运行 data/prepare_mobility.py"
        )

    vehicles = load_stats(stats_path)

    global_first = min(v["first_slot"] for v in vehicles)
    global_last = max(v["last_slot"] for v in vehicles)
    max_length = global_last - global_first + 1

    if capacity_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出已经存在：{capacity_path}\n"
            "如需重新生成，请添加 --overwrite。"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    capacity_rows = []

    # 对所有可能 episode 长度进行分析，不预设 T。
    for T in range(1, max_length + 1):
        start, end, eligible = best_window_for_length(
            vehicles,
            T,
            global_first,
            global_last,
        )

        capacity_rows.append(
            {
                "episode_slots": T,
                "best_start_slot": start,
                "best_end_slot": end,
                "max_eligible_vehicles": len(eligible),
            }
        )

    with capacity_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_slots",
                "best_start_slot",
                "best_end_slot",
                "max_eligible_vehicles",
            ],
        )
        writer.writeheader()
        writer.writerows(capacity_rows)

    print("=" * 76)
    print("Episode window analysis finished")
    print("=" * 76)
    print(f"Vehicles          : {len(vehicles)}")
    print(f"Global slot range : {global_first} ... {global_last}")
    print(f"Global span       : {max_length} slots")
    print(f"Capacity table    : {capacity_path}")

    # 只打印“容量发生变化”的关键点，避免刷屏。
    print("\nCapacity change points:")
    print("-" * 76)
    print(
        f"{'episode_slots':>14} "
        f"{'best_start':>12} "
        f"{'best_end':>10} "
        f"{'eligible':>10}"
    )

    previous_count = None
    for row in capacity_rows:
        count = row["max_eligible_vehicles"]
        if count != previous_count:
            print(
                f"{row['episode_slots']:>14} "
                f"{row['best_start_slot']:>12} "
                f"{row['best_end_slot']:>10} "
                f"{count:>10}"
            )
            previous_count = count

    if args.episode_slots is not None:
        T = args.episode_slots
        start, end, eligible = best_window_for_length(
            vehicles,
            T,
            global_first,
            global_last,
        )

        eligible = sorted(
            eligible,
            key=lambda v: numeric_id_key(v["vehicle_id"])
        )

        eligible_path = (
            processed_dir / f"eligible_vehicles_T{T}.csv"
        )

        if eligible_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"候选车辆文件已经存在：{eligible_path}\n"
                "如需重新生成，请添加 --overwrite。"
            )

        with eligible_path.open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "vehicle_id",
                    "episode_start_slot",
                    "episode_end_slot",
                    "vehicle_first_slot",
                    "vehicle_last_slot",
                ],
            )
            writer.writeheader()

            for v in eligible:
                writer.writerow(
                    {
                        "vehicle_id": v["vehicle_id"],
                        "episode_start_slot": start,
                        "episode_end_slot": end,
                        "vehicle_first_slot": v["first_slot"],
                        "vehicle_last_slot": v["last_slot"],
                    }
                )

        print("\nSelected episode:")
        print("-" * 76)
        print(f"episode_slots     : {T}")
        print(f"start_slot        : {start}")
        print(f"end_slot          : {end}")
        print(f"eligible vehicles : {len(eligible)}")
        print(f"eligible file     : {eligible_path}")

        print("\nVehicle IDs:")
        print(", ".join(v["vehicle_id"] for v in eligible))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
