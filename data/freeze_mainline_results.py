# -*- coding: utf-8 -*-
"""
冻结并汇总完整 0..399 主算法（Proposed/Mainline）结果。

本脚本只做结果归档与统计，不重新运行 P1/P2/Pjoint/P3，
不修改 src/config.py，不修改任何算法参数。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402

SOURCE_FILES = {
    "summary": RESULTS_DIR / "rolling_p1_p2_pjoint_summary_0_399.csv",
    "cache_events": RESULTS_DIR / "rolling_cache_events_0_399.csv",
    "uav_events": RESULTS_DIR / "rolling_uav_events_0_399.csv",
}

CORE_CODE_FILES = [
    SRC_DIR / "config.py",
    SRC_DIR / "channel.py",
    SRC_DIR / "cache_model.py",
    SRC_DIR / "p1_cache.py",
    SRC_DIR / "p2_uav.py",
    SRC_DIR / "pjoint.py",
    SRC_DIR / "p3_solver.py",
    SRC_DIR / "async_update.py",
    SRC_DIR / "scenario_generator.py",
    SRC_DIR / "rolling_controller.py",
    SRC_DIR / "rsu_state.py",
    SRC_DIR / "uav_state.py",
]


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default=0.0) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    return float(default) if text == "" else float(text)


def inum(value, default=0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    return int(default) if text == "" else int(float(text))


def percentile(values: list[float], q: float) -> float | None:
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
    return xs[lo] * (1.0 - w) + xs[hi] * w


def collect_numeric(rows: list[dict], field: str) -> list[float]:
    out = []
    for row in rows:
        if field not in row:
            continue
        text = str(row.get(field, "")).strip()
        if text:
            out.append(float(text))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value) -> str:
    return f"{value:.12g}" if isinstance(value, float) else str(value)


def add_metric(metrics, category, metric, value, unit="", source=""):
    metrics.append({
        "category": category,
        "metric": metric,
        "value": fmt(value),
        "unit": unit,
        "source": source,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有 results/mainline_default。",
    )
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "mainline_default"
    raw_dir = out_dir / "raw"

    if out_dir.exists() and not args.overwrite:
        raise RuntimeError(
            f"{out_dir} 已存在。为避免覆盖已冻结结果，"
            "如确需重建请添加 --overwrite。"
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = load_csv(SOURCE_FILES["summary"])
    cache_events = load_csv(SOURCE_FILES["cache_events"])
    uav_events = load_csv(SOURCE_FILES["uav_events"])

    if len(summary) != 400:
        raise RuntimeError(
            f"主结果应包含 400 个时隙（0..399），实际 {len(summary)} 行。"
        )

    slots = [inum(row.get("slot")) for row in summary]
    if slots != list(range(400)):
        raise RuntimeError(
            "summary 的 slot 序列不是完整 0..399，拒绝冻结。"
        )

    # 冻结原始结果和当前配置。
    for path in SOURCE_FILES.values():
        shutil.copy2(path, raw_dir / path.name)
    shutil.copy2(SRC_DIR / "config.py", out_dir / "config_snapshot.py")

    total_tasks = sum(inum(row.get("task_count")) for row in summary)
    total_cost = sum(fnum(row.get("actual_total_cost")) for row in summary)
    mean_cost = total_cost / total_tasks if total_tasks else 0.0

    mean_delay = (
        sum(
            inum(row.get("task_count"))
            * fnum(row.get("actual_mean_delay_s"))
            for row in summary
        )
        / total_tasks
        if total_tasks else 0.0
    )

    mean_energy_mj = (
        sum(
            inum(row.get("task_count"))
            * fnum(row.get("actual_mean_energy_mJ"))
            for row in summary
        )
        / total_tasks
        if total_tasks else 0.0
    )

    decision_rows = [row for row in summary if inum(row.get("slot")) > 0]
    decision_slots = len(decision_rows)

    cache_changed_slots = sum(inum(r.get("cache_changed")) for r in decision_rows)
    cache_changed_nodes = sum(inum(r.get("cache_changed_nodes")) for r in decision_rows)
    uav_moved_slots = sum(inum(r.get("uav_moved")) for r in decision_rows)
    moved_uavs = sum(inum(r.get("moved_uavs")) for r in decision_rows)
    p1_changed = sum(inum(r.get("p1_changed")) for r in decision_rows)
    p2_changed = sum(inum(r.get("p2_changed")) for r in decision_rows)
    pjoint_checked = sum(inum(r.get("pjoint_checked")) for r in decision_rows)
    pjoint_changed = sum(inum(r.get("pjoint_changed")) for r in decision_rows)

    mode_fields = {
        "SV": "sv_tasks",
        "RSU": "rsu_tasks",
        "UAV": "uav_tasks",
        "CLOUD": "cloud_tasks",
    }
    mode_counts = {}
    for mode, field in mode_fields.items():
        if field in summary[0]:
            mode_counts[mode] = sum(inum(r.get(field)) for r in summary)
        else:
            mode_counts[mode] = None

    known_mode_sum = sum(v for v in mode_counts.values() if v is not None)

    cache_intervals = collect_numeric(cache_events, "update_interval_slots")
    uav_intervals = collect_numeric(uav_events, "update_interval_slots")
    uav_distances = collect_numeric(uav_events, "distance_m")

    metrics = []
    add_metric(metrics, "episode", "slots_total", 400, "slot", "summary")
    add_metric(metrics, "episode", "decision_slots", decision_slots, "slot", "summary")
    add_metric(metrics, "performance", "tasks_total", total_tasks, "task", "summary")
    add_metric(metrics, "performance", "mean_normalized_cost", mean_cost, "", "summary")
    add_metric(metrics, "performance", "mean_delay_s", mean_delay, "s", "summary")
    add_metric(metrics, "performance", "mean_energy_mJ", mean_energy_mj, "mJ", "summary")

    for mode in ("SV", "RSU", "UAV", "CLOUD"):
        count = mode_counts[mode]
        if count is None:
            continue
        add_metric(metrics, "execution_mode", f"{mode.lower()}_tasks", count, "task", "summary")
        add_metric(
            metrics,
            "execution_mode",
            f"{mode.lower()}_task_ratio",
            count / total_tasks if total_tasks else 0.0,
            "ratio",
            "summary",
        )

    add_metric(metrics, "async_update", "cache_update_slots", cache_changed_slots, "slot", "summary")
    add_metric(metrics, "async_update", "cache_update_slot_ratio", cache_changed_slots / decision_slots, "ratio", "summary")
    add_metric(metrics, "async_update", "cache_node_events", cache_changed_nodes, "event", "summary")
    add_metric(metrics, "async_update", "uav_move_slots", uav_moved_slots, "slot", "summary")
    add_metric(metrics, "async_update", "uav_move_slot_ratio", uav_moved_slots / decision_slots, "ratio", "summary")
    add_metric(metrics, "async_update", "uav_move_events", moved_uavs, "event", "summary")
    add_metric(metrics, "algorithm", "p1_changed_slots", p1_changed, "slot", "summary")
    add_metric(metrics, "algorithm", "p2_changed_slots", p2_changed, "slot", "summary")
    add_metric(metrics, "algorithm", "pjoint_checked_slots", pjoint_checked, "slot", "summary")
    add_metric(metrics, "algorithm", "pjoint_changed_slots", pjoint_changed, "slot", "summary")

    for prefix, values, unit in [
        ("cache_update_interval", cache_intervals, "slot"),
        ("uav_update_interval", uav_intervals, "slot"),
        ("uav_move_distance", uav_distances, "m"),
    ]:
        for label, q in [("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90)]:
            value = percentile(values, q)
            if value is not None:
                add_metric(metrics, "distribution", f"{prefix}_{label}", value, unit, "event_csv")

    config_names = [
        "CLOUD_PSI",
        "CACHE_FIXED_SWITCH_COST",
        "CACHE_SIZE_COST_COEF",
        "UAV_FIXED_SWITCH_COST",
        "UAV_MOVE_COST_COEF",
        "PREDICTION_HORIZON",
        "NUM_SCENARIOS",
        "NUM_TV",
        "NUM_SV",
        "NUM_SERVICES",
    ]
    for name in config_names:
        if hasattr(config, name):
            add_metric(metrics, "config", name, getattr(config, name), "", "src/config.py")

    metrics_file = out_dir / "mainline_metrics.csv"
    with metrics_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "metric", "value", "unit", "source"],
        )
        writer.writeheader()
        writer.writerows(metrics)

    lines = [
        "=" * 78,
        "完整 400-slot Proposed / Mainline 结果快照",
        "=" * 78,
        f"生成时间                  : {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "时隙范围                  : 0 ... 399",
        f"实际决策时隙数            : {decision_slots}",
        f"实际任务总数              : {total_tasks}",
        f"实际任务平均归一化代价    : {mean_cost:.9f}",
        f"实际任务平均时延          : {mean_delay:.9f} s",
        f"实际任务平均能耗          : {mean_energy_mj:.9f} mJ",
        "",
        "执行模式：",
    ]

    for mode in ("SV", "RSU", "UAV", "CLOUD"):
        count = mode_counts[mode]
        if count is None:
            lines.append(
                f"  {mode:<5}                    : 当前 summary 无对应字段，未凭空记为 0"
            )
        else:
            ratio = 100.0 * count / total_tasks if total_tasks else 0.0
            lines.append(f"  {mode:<5}                    : {count}/{total_tasks} ({ratio:.2f}%)")

    lines += [
        "",
        f"已识别执行模式任务合计    : {known_mode_sum}/{total_tasks}",
        "",
        "异步更新：",
        f"  缓存更新时隙            : {cache_changed_slots}/{decision_slots} ({100*cache_changed_slots/decision_slots:.2f}%)",
        f"  节点级缓存更新事件      : {cache_changed_nodes}",
        f"  UAV 移动时隙            : {uav_moved_slots}/{decision_slots} ({100*uav_moved_slots/decision_slots:.2f}%)",
        f"  UAV 移动事件            : {moved_uavs}",
        "",
        "算法动作：",
        f"  P1 changed              : {p1_changed}/{decision_slots}",
        f"  P2 changed              : {p2_changed}/{decision_slots}",
        f"  Pjoint checked          : {pjoint_checked}/{decision_slots}",
        f"  Pjoint changed          : {pjoint_changed}/{decision_slots}",
        "",
    ]

    if cache_intervals:
        lines.append(
            "Cache update interval      : "
            f"P50={percentile(cache_intervals, .50):.3f}, "
            f"P75={percentile(cache_intervals, .75):.3f}, "
            f"P90={percentile(cache_intervals, .90):.3f} slot"
        )
    if uav_intervals:
        lines.append(
            "UAV update interval        : "
            f"P50={percentile(uav_intervals, .50):.3f}, "
            f"P75={percentile(uav_intervals, .75):.3f}, "
            f"P90={percentile(uav_intervals, .90):.3f} slot"
        )
    if uav_distances:
        lines.append(
            "UAV movement distance      : "
            f"P50={percentile(uav_distances, .50):.3f}, "
            f"P75={percentile(uav_distances, .75):.3f}, "
            f"P90={percentile(uav_distances, .90):.3f} m"
        )

    lines += ["", "当前冻结参数："]
    for name in config_names:
        if hasattr(config, name):
            lines.append(f"  {name:<26}: {getattr(config, name)}")

    lines += [
        "",
        "说明：",
        "- 本目录是当前 Proposed/Mainline 的正式快照。",
        "- 后续 baseline、消融、参数敏感性统一以 mainline_metrics.csv 为参照。",
        "- 本脚本未重新运行求解器，也未修改任何算法参数。",
    ]

    report_file = out_dir / "mainline_report.txt"
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_rows = []
    snapshot_files = [
        raw_dir / SOURCE_FILES["summary"].name,
        raw_dir / SOURCE_FILES["cache_events"].name,
        raw_dir / SOURCE_FILES["uav_events"].name,
        out_dir / "config_snapshot.py",
        metrics_file,
        report_file,
    ]
    for path in snapshot_files:
        manifest_rows.append({
            "type": "snapshot",
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        })
    for path in CORE_CODE_FILES:
        if path.is_file():
            manifest_rows.append({
                "type": "core_code",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            })

    manifest_file = out_dir / "manifest_sha256.csv"
    with manifest_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "path", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n".join(lines))
    print()
    print("=" * 78)
    print("主算法结果已冻结")
    print("=" * 78)
    print(f"目录      : {out_dir}")
    print(f"指标表    : {metrics_file}")
    print(f"摘要      : {report_file}")
    print(f"SHA256    : {manifest_file}")


if __name__ == "__main__":
    main()
