# -*- coding: utf-8 -*-
"""
配置更新动态诊断。

用途：
读取已经完成的 rolling_controller 结果，
分析当前 P1 / P2 / Pjoint / 异步更新行为。

本脚本：
- 不修改 config.py；
- 不重新运行 P1/P2/P3；
- 不生成新的任务或轨迹；
- 只分析已有 CSV。

重点回答：
1. P1 每次接受的收益量级；
2. P2 每次接受的收益量级；
3. UAV 实际移动距离量级；
4. 缓存与 UAV 的实际更新间隔；
5. 当前 0 switching-cost 下是否出现“几乎每时隙更新”。

默认分析：
    results/rolling_p1_p2_pjoint_summary_0_20.csv
    results/rolling_cache_events_0_20.csv
    results/rolling_uav_events_0_20.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def to_float(value) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return float(text)


def to_int(value) -> int:
    if value is None:
        return 0

    text = str(value).strip()

    if text == "":
        return 0

    return int(float(text))


def percentile(
    values: list[float],
    q: float,
) -> float | None:
    if not values:
        return None

    xs = sorted(values)

    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    w = pos - lo

    return (
        xs[lo] * (1.0 - w)
        + xs[hi] * w
    )


def print_distribution(
    title: str,
    values: list[float],
    unit: str = "",
) -> None:
    if not values:
        print(
            f"{title:<30}: 无数据"
        )
        return

    p10 = percentile(values, 0.10)
    p25 = percentile(values, 0.25)
    p50 = percentile(values, 0.50)
    p75 = percentile(values, 0.75)
    p90 = percentile(values, 0.90)

    print(
        f"{title:<30}: "
        f"P10={p10:.9g}, "
        f"P25={p25:.9g}, "
        f"P50={p50:.9g}, "
        f"P75={p75:.9g}, "
        f"P90={p90:.9g}"
        + (
            f" {unit}"
            if unit
            else ""
        )
    )


def collect_float(
    rows: list[dict],
    field: str,
    require_field: str | None = None,
) -> list[float]:
    result = []

    for row in rows:
        if (
            require_field is not None
            and to_int(
                row.get(require_field)
            ) == 0
        ):
            continue

        value = to_float(
            row.get(field)
        )

        if value is not None:
            result.append(value)

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--end-slot",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    end_slot = args.end_slot

    summary_file = (
        RESULTS_DIR
        / (
            "rolling_p1_p2_pjoint_summary_"
            f"0_{end_slot}.csv"
        )
    )

    cache_file = (
        RESULTS_DIR
        / (
            "rolling_cache_events_"
            f"0_{end_slot}.csv"
        )
    )

    uav_file = (
        RESULTS_DIR
        / (
            "rolling_uav_events_"
            f"0_{end_slot}.csv"
        )
    )

    summary = load_csv(
        summary_file
    )

    cache_events = load_csv(
        cache_file
    )

    uav_events = load_csv(
        uav_file
    )

    if not summary:
        raise RuntimeError(
            "rolling summary 为空。"
        )

    decision_rows = [
        row
        for row in summary
        if to_int(
            row.get("slot")
        ) > 0
    ]

    decision_slots = len(
        decision_rows
    )

    p1_changed = sum(
        to_int(
            row.get("p1_changed")
        )
        for row in decision_rows
    )

    p2_changed = sum(
        to_int(
            row.get("p2_changed")
        )
        for row in decision_rows
    )

    pjoint_checked = sum(
        to_int(
            row.get("pjoint_checked")
        )
        for row in decision_rows
    )

    pjoint_changed = sum(
        to_int(
            row.get("pjoint_changed")
        )
        for row in decision_rows
    )

    cache_changed_slots = sum(
        to_int(
            row.get("cache_changed")
        )
        for row in decision_rows
    )

    uav_moved_slots = sum(
        to_int(
            row.get("uav_moved")
        )
        for row in decision_rows
    )

    p1_gross = collect_float(
        decision_rows,
        "p1_gross_gain",
        "p1_changed",
    )

    p1_net = collect_float(
        decision_rows,
        "p1_net_gain",
        "p1_changed",
    )

    p1_switch = collect_float(
        decision_rows,
        "p1_switch_cost",
        "p1_changed",
    )

    p2_gross = collect_float(
        decision_rows,
        "p2_gross_gain",
        "p2_changed",
    )

    p2_net = collect_float(
        decision_rows,
        "p2_net_gain",
        "p2_changed",
    )

    p2_move_cost = collect_float(
        decision_rows,
        "p2_move_cost",
        "p2_changed",
    )

    p1_eval = collect_float(
        decision_rows,
        "p1_p3_evaluations",
    )

    p2_eval = collect_float(
        decision_rows,
        "p2_p3_evaluations",
    )

    uav_distances = collect_float(
        uav_events,
        "distance_m",
    )

    uav_intervals = collect_float(
        uav_events,
        "update_interval_slots",
    )

    cache_intervals = collect_float(
        cache_events,
        "update_interval_slots",
    )

    # 当前只有 1 UAV 时，这个值具有直接解释：
    # 每米移动成本系数达到 gross_gain/distance 后，
    # 该次移动会来到 break-even 附近。
    # 这里只输出诊断尺度，不自动推荐参数。
    uav_break_even_per_m = []

    gross_by_slot = {}

    for row in decision_rows:
        if to_int(
            row.get("p2_changed")
        ) == 0:
            continue

        slot = to_int(
            row.get("slot")
        )

        gross = to_float(
            row.get(
                "p2_gross_gain"
            )
        )

        if gross is not None:
            gross_by_slot[slot] = gross

    for event in uav_events:
        slot = to_int(
            event.get("slot")
        )

        distance = to_float(
            event.get(
                "distance_m"
            )
        )

        gross = gross_by_slot.get(
            slot
        )

        if (
            gross is not None
            and distance is not None
            and distance > 1e-12
        ):
            uav_break_even_per_m.append(
                gross / distance
            )

    print("=" * 78)
    print(
        "P1 / P2 / Pjoint 配置更新动态诊断"
    )
    print("=" * 78)

    print(
        f"分析时隙                  : "
        f"0 ... {end_slot}"
    )

    print(
        f"实际决策时隙数            : "
        f"{decision_slots}"
    )

    print()

    print("A. 更新频率")
    print(
        f"  P1 changed              : "
        f"{p1_changed}/{decision_slots} "
        f"({100*p1_changed/decision_slots:.2f}%)"
    )
    print(
        f"  P2 changed              : "
        f"{p2_changed}/{decision_slots} "
        f"({100*p2_changed/decision_slots:.2f}%)"
    )
    print(
        f"  最终 cache_changed      : "
        f"{cache_changed_slots}/{decision_slots} "
        f"({100*cache_changed_slots/decision_slots:.2f}%)"
    )
    print(
        f"  最终 uav_moved          : "
        f"{uav_moved_slots}/{decision_slots} "
        f"({100*uav_moved_slots/decision_slots:.2f}%)"
    )
    print(
        f"  Pjoint checked          : "
        f"{pjoint_checked}/{decision_slots}"
    )
    print(
        f"  Pjoint changed          : "
        f"{pjoint_changed}/{decision_slots}"
    )

    print()
    print("B. P1 收益与切换成本")
    print_distribution(
        "  P1 gross gain",
        p1_gross,
    )
    print_distribution(
        "  P1 net gain",
        p1_net,
    )
    print_distribution(
        "  P1 switch cost",
        p1_switch,
    )

    print()
    print("C. P2 收益与移动成本")
    print_distribution(
        "  P2 gross gain",
        p2_gross,
    )
    print_distribution(
        "  P2 net gain",
        p2_net,
    )
    print_distribution(
        "  P2 move cost",
        p2_move_cost,
    )

    print()
    print("D. UAV 实际移动")
    print_distribution(
        "  movement distance",
        uav_distances,
        "m",
    )
    print_distribution(
        "  update interval",
        uav_intervals,
        "slot",
    )
    print_distribution(
        "  gross gain / distance",
        uav_break_even_per_m,
        "cost/m",
    )

    print()
    print("E. Cache 实际更新")
    print(
        f"  节点级 cache events      : "
        f"{len(cache_events)}"
    )
    print_distribution(
        "  cache update interval",
        cache_intervals,
        "slot",
    )

    print()
    print("F. 求解评价次数")
    print_distribution(
        "  P1 P3 evaluations/slot",
        p1_eval,
    )
    print_distribution(
        "  P2 P3 evaluations/slot",
        p2_eval,
    )

    print()
    print("解释：")
    print(
        "- 若 P2 changed 和 uav_moved 接近 100%，"
        "说明当前 UAV 更新抑制机制过弱。"
    )
    print(
        "- 若 move cost 全为 0，而 gross gain 很小，"
        "则任何微小严格改善都会触发真实移动。"
    )
    print(
        "- gross gain / distance 只给出当前数据下的"
        "移动成本量级参考，不自动修改 UAV_MOVE_COST_COEF。"
    )
    print(
        "- Cache 同理；先观察实际收益和更新间隔，"
        "再单独校准缓存切换成本。"
    )

    print()
    print(
        f"summary                  : "
        f"{summary_file}"
    )
    print(
        f"cache events             : "
        f"{cache_file}"
    )
    print(
        f"uav events               : "
        f"{uav_file}"
    )


if __name__ == "__main__":
    main()
