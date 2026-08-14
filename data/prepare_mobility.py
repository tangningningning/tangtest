#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 SUMO traceExporter.py 的 GPSDAT 输出整理为本项目统一 mobility trace。

放置位置：
    adaptive_cache_mec/data/prepare_mobility.py

默认输入：
    data/sumo/output/vehicle_trace_1s.csv

默认输出：
    data/sumo/processed/mobility_trace.csv
    data/sumo/processed/vehicle_trajectory_stats.csv

统一轨迹格式：
    slot,time_s,vehicle_id,x,y,speed_mps

说明：
1. 输入文件虽然扩展名为 .csv，但 SUMO GPSDAT 实际使用 TAB 分隔。
2. GPSDAT 速度为 km/h，本脚本转换为 m/s。
3. time_s 恢复为相对于当天 00:00:00 的 SUMO 仿真时间。
4. vehicle_id 始终按字符串保存，不擅自改编号。
5. 本脚本只做轨迹标准化和统计，不选择 TV/SV。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Record:
    vehicle_id: str
    timestamp: datetime
    x: float
    y: float
    speed_kmh: float


def project_root_from_script() -> Path:
    """
    预期脚本位置为 <project>/data/prepare_mobility.py。
    """
    script_path = Path(__file__).resolve()

    if script_path.parent.name == "data":
        root = script_path.parent.parent
        if (root / "data" / "sumo").is_dir():
            return root

    cwd = Path.cwd().resolve()
    if (cwd / "data" / "sumo").is_dir():
        return cwd

    raise FileNotFoundError(
        "无法定位项目根目录。请确认脚本位于项目 data/ 目录，"
        "或者从项目根目录运行。"
    )


def parse_gpsdat(path: Path) -> list[Record]:
    """
    SUMO GPSDAT 每行格式：
        vehicle_id<TAB>YYYY-MM-DD HH:MM:SS<TAB>x<TAB>y<TAB>status<TAB>speed_kmh
    """
    records: list[Record] = []

    with path.open("r", encoding="utf-8", errors="strict", newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        for line_no, row in enumerate(reader, start=1):
            # 跳过空行
            if not row or all(not item.strip() for item in row):
                continue

            row = [item.strip() for item in row]

            if len(row) != 6:
                raise ValueError(
                    f"第 {line_no} 行字段数不是 6：{row}\n"
                    "请确认输入文件是 SUMO traceExporter 的 GPSDAT 输出。"
                )

            vehicle_id, timestamp_text, x_text, y_text, _status, speed_text = row

            try:
                timestamp = datetime.strptime(
                    timestamp_text, "%Y-%m-%d %H:%M:%S"
                )
                x = float(x_text)
                y = float(y_text)
                speed_kmh = float(speed_text)
            except ValueError as exc:
                raise ValueError(
                    f"第 {line_no} 行解析失败：{row}"
                ) from exc

            records.append(
                Record(
                    vehicle_id=vehicle_id,
                    timestamp=timestamp,
                    x=x,
                    y=y,
                    speed_kmh=speed_kmh,
                )
            )

    if not records:
        raise ValueError(f"输入轨迹为空：{path}")

    return records


def simulation_time_seconds(timestamp: datetime, base_midnight: datetime) -> float:
    return (timestamp - base_midnight).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="标准化 SUMO 1 秒车辆轨迹并生成车辆轨迹统计。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="输入 GPSDAT 文件；默认 data/sumo/output/vehicle_trace_1s.csv",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=1.0,
        help="MEC 决策槽长度（秒），默认 1.0。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )
    args = parser.parse_args()

    if args.delta_t <= 0:
        raise ValueError("--delta-t 必须大于 0。")

    root = project_root_from_script()

    input_path = (
        args.input
        if args.input is not None
        else root / "data" / "sumo" / "output" / "vehicle_trace_1s.csv"
    )
    if not input_path.is_absolute():
        input_path = (root / input_path).resolve()
    else:
        input_path = input_path.resolve()

    output_dir = root / "data" / "sumo" / "processed"
    trace_output = output_dir / "mobility_trace.csv"
    stats_output = output_dir / "vehicle_trajectory_stats.csv"

    if not input_path.is_file():
        raise FileNotFoundError(f"输入轨迹不存在：{input_path}")

    existing = [p for p in (trace_output, stats_output) if p.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "以下输出文件已经存在：\n"
            f"{paths}\n"
            "如需重新生成，请添加 --overwrite。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_gpsdat(input_path)

    # GPSDAT 的日期只是 traceExporter 使用的基准日期。
    # 将最早记录所在日期的 00:00:00 作为仿真时间 0 的基准。
    earliest_timestamp = min(r.timestamp for r in records)
    base_midnight = earliest_timestamp.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    normalized_rows = []

    for r in records:
        time_s = simulation_time_seconds(r.timestamp, base_midnight)

        # 由于当前轨迹已经按 1 s 导出，这里应接近整数 slot。
        raw_slot = time_s / args.delta_t
        slot = int(round(raw_slot))

        if abs(raw_slot - slot) > 1e-6:
            raise ValueError(
                f"时间 {time_s} s 无法准确映射到 delta_t={args.delta_t} s 的 slot。"
            )

        normalized_rows.append(
            {
                "slot": slot,
                "time_s": time_s,
                "vehicle_id": r.vehicle_id,
                "x": r.x,
                "y": r.y,
                "speed_mps": r.speed_kmh / 3.6,
            }
        )

    # 稳定排序：先按 slot，再按 vehicle_id 的字符串形式。
    normalized_rows.sort(
        key=lambda row: (row["slot"], row["vehicle_id"])
    )

    # 检查同一车辆同一 slot 是否重复。
    seen = set()
    for row in normalized_rows:
        key = (row["vehicle_id"], row["slot"])
        if key in seen:
            raise ValueError(
                f"发现重复车辆-slot 记录：vehicle_id={key[0]}, slot={key[1]}"
            )
        seen.add(key)

    # 写标准 mobility trace。
    with trace_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slot",
                "time_s",
                "vehicle_id",
                "x",
                "y",
                "speed_mps",
            ],
        )
        writer.writeheader()

        for row in normalized_rows:
            writer.writerow(
                {
                    "slot": row["slot"],
                    "time_s": f'{row["time_s"]:.1f}',
                    "vehicle_id": row["vehicle_id"],
                    "x": f'{row["x"]:.2f}',
                    "y": f'{row["y"]:.2f}',
                    "speed_mps": f'{row["speed_mps"]:.6f}',
                }
            )

    # 按车辆统计轨迹完整性。
    vehicle_slots: dict[str, list[int]] = {}
    for row in normalized_rows:
        vehicle_slots.setdefault(row["vehicle_id"], []).append(row["slot"])

    stats_rows = []

    for vehicle_id, slots in vehicle_slots.items():
        slots = sorted(slots)

        first_slot = slots[0]
        last_slot = slots[-1]
        num_records = len(slots)

        expected_records = last_slot - first_slot + 1
        missing_slots = expected_records - num_records
        coverage_ratio = (
            num_records / expected_records if expected_records > 0 else 0.0
        )
        continuous = missing_slots == 0

        stats_rows.append(
            {
                "vehicle_id": vehicle_id,
                "first_slot": first_slot,
                "last_slot": last_slot,
                "num_records": num_records,
                "expected_records": expected_records,
                "missing_slots": missing_slots,
                "coverage_ratio": coverage_ratio,
                "continuous": continuous,
                "duration_s": (last_slot - first_slot) * args.delta_t,
            }
        )

    stats_rows.sort(
        key=lambda row: (
            -row["num_records"],
            row["vehicle_id"],
        )
    )

    with stats_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vehicle_id",
                "first_slot",
                "last_slot",
                "num_records",
                "expected_records",
                "missing_slots",
                "coverage_ratio",
                "continuous",
                "duration_s",
            ],
        )
        writer.writeheader()

        for row in stats_rows:
            writer.writerow(
                {
                    "vehicle_id": row["vehicle_id"],
                    "first_slot": row["first_slot"],
                    "last_slot": row["last_slot"],
                    "num_records": row["num_records"],
                    "expected_records": row["expected_records"],
                    "missing_slots": row["missing_slots"],
                    "coverage_ratio": f'{row["coverage_ratio"]:.6f}',
                    "continuous": row["continuous"],
                    "duration_s": f'{row["duration_s"]:.1f}',
                }
            )

    all_slots = [row["slot"] for row in normalized_rows]
    continuous_count = sum(1 for row in stats_rows if row["continuous"])

    print("=" * 72)
    print("Mobility preprocessing finished")
    print("=" * 72)
    print(f"Input              : {input_path}")
    print(f"Mobility trace     : {trace_output}")
    print(f"Trajectory stats   : {stats_output}")
    print(f"Decision delta_t   : {args.delta_t} s")
    print(f"Records            : {len(normalized_rows)}")
    print(f"Vehicles           : {len(vehicle_slots)}")
    print(f"Global slot range  : {min(all_slots)} ... {max(all_slots)}")
    print(f"Continuous vehicles: {continuous_count}/{len(vehicle_slots)}")

    print("\nTop 10 vehicles by trajectory length:")
    print("-" * 72)
    print(
        f"{'vehicle_id':>12} "
        f"{'first':>7} "
        f"{'last':>7} "
        f"{'records':>8} "
        f"{'missing':>8} "
        f"{'coverage':>10}"
    )

    for row in stats_rows[:10]:
        print(
            f"{row['vehicle_id']:>12} "
            f"{row['first_slot']:>7} "
            f"{row['last_slot']:>7} "
            f"{row['num_records']:>8} "
            f"{row['missing_slots']:>8} "
            f"{row['coverage_ratio']:>10.4f}"
        )

    print("\nFirst 5 normalized rows:")
    print("-" * 72)
    for row in normalized_rows[:5]:
        print(
            f"slot={row['slot']}, "
            f"time_s={row['time_s']:.1f}, "
            f"vehicle_id={row['vehicle_id']}, "
            f"x={row['x']:.2f}, "
            f"y={row['y']:.2f}, "
            f"speed_mps={row['speed_mps']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
