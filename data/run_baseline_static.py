# -*- coding: utf-8 -*-
"""
Baseline 1：Static-Config（静态配置）完整 400-slot 实验。

公平性原则：
1. 与 Proposed 使用完全相同的真实 mobility、task、TV/SV/RSU/UAV/Cloud 参数；
2. t=0 使用与 rolling_controller 完全相同的 UAV 初始化；
3. t=0 使用与 rolling_controller 完全相同的初始缓存状态；
4. 从 t=1 开始不再运行 P1/P2/Pjoint，不更新缓存、不移动 UAV；
5. 每个真实时隙仍调用同一个正式 P3，执行节点选择与 CPU/带宽分配不变；
6. SV 仍按 SUMO 实际轨迹移动，RSU 仍固定，Cloud 不变；
7. 不修改 src/config.py，不覆盖 Proposed 结果。

该 baseline 只去掉“跨时隙配置自适应”，用于直接衡量：
    P1 + P2 + Pjoint + Async
相对于“初始配置一直保持不变”的收益。

输出：
    results/baseline_static/
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cache_model  # noqa: E402
import config  # noqa: E402
import p3_solver  # noqa: E402
import rolling_controller  # noqa: E402
import uav_state  # noqa: E402


END_SLOT = 399
OUT_DIR = RESULTS_DIR / "baseline_static"
SUMMARY_FILE = OUT_DIR / "baseline_static_summary_0_399.csv"
METRICS_FILE = OUT_DIR / "baseline_static_metrics.csv"
REPORT_FILE = OUT_DIR / "baseline_static_report.txt"
MANIFEST_FILE = OUT_DIR / "manifest_sha256.csv"

PROPOSED_SUMMARY = (
    RESULTS_DIR
    / "mainline_default"
    / "raw"
    / "rolling_p1_p2_pjoint_summary_0_399.csv"
)


def group_tasks_by_slot(tasks: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for task in tasks:
        grouped[int(task["slot"])].append(task)
    return dict(grouped)


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def proposed_realized_metrics(rows: list[dict]) -> dict:
    """
    从已冻结 Proposed summary 计算真实 realized objective。

    当前 summary 的 actual_total_cost 是真实任务 P3 代价；
    配置更新成本另存在 P1/P2/Pjoint 字段中。

    若 Pjoint 最终接受：
        使用 Pjoint 的 cache + UAV switching cost。
    否则：
        使用本时隙实际接受的 P1 switch cost + P2 move cost。

    t=0 为外生初始状态，不计初始化成本。
    """
    total_tasks = 0
    task_cost = 0.0
    config_cost = 0.0

    for row in rows:
        total_tasks += inum(row.get("task_count"))
        task_cost += fnum(row.get("actual_total_cost"))

        if inum(row.get("slot")) <= 0:
            continue

        if inum(row.get("pjoint_changed")):
            config_cost += fnum(row.get("pjoint_cache_switch_cost"))
            config_cost += fnum(row.get("pjoint_uav_move_cost"))
        else:
            if inum(row.get("p1_changed")):
                config_cost += fnum(row.get("p1_switch_cost"))
            if inum(row.get("p2_changed")):
                config_cost += fnum(row.get("p2_move_cost"))

    total_objective = task_cost + config_cost

    return {
        "tasks": total_tasks,
        "task_cost": task_cost,
        "config_cost": config_cost,
        "total_objective": total_objective,
        "mean_task_cost": task_cost / total_tasks,
        "mean_total_objective": total_objective / total_tasks,
    }


def main():
    if config.T != 400:
        raise RuntimeError(
            f"当前 baseline 固定对应完整 400-slot episode，"
            f"但 config.T={config.T}。"
        )

    if not PROPOSED_SUMMARY.is_file():
        raise FileNotFoundError(
            "缺少已冻结 Proposed summary：\n"
            f"{PROPOSED_SUMMARY}\n"
            "请先完成 results/mainline_default 冻结。"
        )

    if OUT_DIR.exists():
        raise RuntimeError(
            f"{OUT_DIR} 已存在。为避免覆盖 baseline 正式结果，"
            "请先确认旧目录后再手动删除。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # ==============================================================
    # 1. 加载与 Proposed 完全相同的外生数据
    # ==============================================================
    actual_positions = p3_solver.load_positions()
    all_tasks = p3_solver.load_tasks()
    tasks_by_slot = group_tasks_by_slot(all_tasks)

    tv_ids = p3_solver.load_ranked_vehicle_ids(
        "TV",
        config.NUM_TV,
    )

    # ==============================================================
    # 2. 完全复用 Proposed 的 t=0 初始化
    # ==============================================================
    static_uav_positions = uav_state.initialize_uav_positions(
        tv_ids=tv_ids,
        actual_positions=actual_positions,
        num_uav=config.NUM_UAV,
        initial_slot=0,
    )

    static_edge_nodes = p3_solver.build_edge_nodes(
        static_uav_positions
    )

    static_cache = cache_model.build_initial_cache_state(
        static_edge_nodes
    )

    # 从此以后：
    # static_uav_positions 不再改变；
    # static_cache 不再改变。
    # 但 SV 的实际位置仍由 P3 按各 slot 的 SUMO mobility 读取。
    rows = []

    total_tasks = 0
    total_cost = 0.0
    total_delay = 0.0
    total_energy_mj = 0.0

    mode_totals = {
        "SV": 0,
        "RSU": 0,
        "UAV": 0,
        "CLOUD": 0,
    }

    for slot in range(END_SLOT + 1):
        slot_tasks = tasks_by_slot.get(slot, [])

        actual = rolling_controller.run_actual_slot_p3(
            slot_tasks=slot_tasks,
            edge_nodes=static_edge_nodes,
            positions=actual_positions,
            cache_state=static_cache,
        )

        n = int(actual["task_count"])
        total_tasks += n
        total_cost += float(actual["total_cost"])
        total_delay += n * float(actual["mean_delay"])
        total_energy_mj += n * float(actual["mean_energy_mJ"])

        mode_counts = actual["mode_counts"]

        for mode in mode_totals:
            mode_totals[mode] += int(
                mode_counts.get(mode, 0)
            )

        rows.append(
            {
                "slot": slot,
                "task_count": n,
                "edge_eligible_tasks": int(actual["eligible_tasks"]),
                "sv_tasks": int(mode_counts.get("SV", 0)),
                "rsu_tasks": int(mode_counts.get("RSU", 0)),
                "uav_tasks": int(mode_counts.get("UAV", 0)),
                "cloud_tasks": int(mode_counts.get("CLOUD", 0)),
                "actual_total_cost": f"{float(actual['total_cost']):.12f}",
                "actual_mean_cost": f"{float(actual['mean_cost']):.12f}",
                "actual_mean_delay_s": f"{float(actual['mean_delay']):.12f}",
                "actual_mean_energy_mJ": f"{float(actual['mean_energy_mJ']):.12f}",
                "cache_changed": 0,
                "uav_moved": 0,
            }
        )

        if slot % 10 == 0 or slot == END_SLOT:
            print(
                f"[Static slot {slot:03d}] "
                f"tasks={n}, "
                f"SV={mode_counts.get('SV', 0)}, "
                f"RSU={mode_counts.get('RSU', 0)}, "
                f"UAV={mode_counts.get('UAV', 0)}, "
                f"Cloud={mode_counts.get('CLOUD', 0)}"
            )

    if total_tasks != len(all_tasks):
        raise RuntimeError(
            f"Static baseline 实际处理任务 {total_tasks}，"
            f"但任务文件共有 {len(all_tasks)}。"
        )

    if sum(mode_totals.values()) != total_tasks:
        raise RuntimeError(
            "执行模式计数之和与任务总数不一致。"
        )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    static_mean_cost = total_cost / total_tasks
    static_mean_delay = total_delay / total_tasks
    static_mean_energy_mj = total_energy_mj / total_tasks

    # Static 没有配置更新，所以真实配置成本为 0。
    static_config_cost = 0.0
    static_total_objective = total_cost
    static_mean_total_objective = static_total_objective / total_tasks

    # ==============================================================
    # 3. 与已冻结 Proposed 做同口径比较
    # ==============================================================
    proposed_rows = load_csv(PROPOSED_SUMMARY)
    proposed = proposed_realized_metrics(proposed_rows)

    if proposed["tasks"] != total_tasks:
        raise RuntimeError(
            "Proposed 与 Static 的任务总数不同，拒绝生成比较结果。"
        )

    task_cost_change = (
        (static_mean_cost - proposed["mean_task_cost"])
        / proposed["mean_task_cost"]
    )

    objective_change = (
        (static_mean_total_objective - proposed["mean_total_objective"])
        / proposed["mean_total_objective"]
    )

    metrics = [
        ("baseline", "Static-Config", ""),
        ("slots_total", 400, "slot"),
        ("decision_slots", 399, "slot"),
        ("tasks_total", total_tasks, "task"),
        ("task_cost_total", total_cost, "cost"),
        ("config_update_cost_total", static_config_cost, "cost"),
        ("realized_total_objective", static_total_objective, "cost"),
        ("mean_normalized_task_cost", static_mean_cost, "cost/task"),
        ("mean_realized_total_objective", static_mean_total_objective, "cost/task"),
        ("mean_delay_s", static_mean_delay, "s"),
        ("mean_energy_mJ", static_mean_energy_mj, "mJ"),
        ("sv_tasks", mode_totals["SV"], "task"),
        ("rsu_tasks", mode_totals["RSU"], "task"),
        ("uav_tasks", mode_totals["UAV"], "task"),
        ("cloud_tasks", mode_totals["CLOUD"], "task"),
        ("cache_update_slots", 0, "slot"),
        ("uav_move_slots", 0, "slot"),
        ("vs_proposed_task_cost_change", task_cost_change, "ratio"),
        ("vs_proposed_total_objective_change", objective_change, "ratio"),
        ("proposed_mean_task_cost", proposed["mean_task_cost"], "cost/task"),
        ("proposed_config_update_cost_total", proposed["config_cost"], "cost"),
        ("proposed_mean_total_objective", proposed["mean_total_objective"], "cost/task"),
    ]

    with METRICS_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit"])
        for metric, value, unit in metrics:
            if isinstance(value, float):
                value = f"{value:.12g}"
            writer.writerow([metric, value, unit])

    lines = [
        "=" * 78,
        "Baseline 1：Static-Config 完整 400-slot 结果",
        "=" * 78,
        f"生成时间                  : {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "定义                      : t=0 初始缓存/UAV位置后永久冻结；每时隙仍精确求 P3",
        "时隙范围                  : 0 ... 399",
        f"实际任务总数              : {total_tasks}",
        "",
        "Static-Config：",
        f"  平均归一化任务代价      : {static_mean_cost:.9f}",
        f"  配置更新总成本          : {static_config_cost:.9f}",
        f"  平均 realized 总目标    : {static_mean_total_objective:.9f}",
        f"  平均时延                : {static_mean_delay:.9f} s",
        f"  平均能耗                : {static_mean_energy_mj:.9f} mJ",
        f"  SV                      : {mode_totals['SV']}/{total_tasks} ({100*mode_totals['SV']/total_tasks:.2f}%)",
        f"  RSU                     : {mode_totals['RSU']}/{total_tasks} ({100*mode_totals['RSU']/total_tasks:.2f}%)",
        f"  UAV                     : {mode_totals['UAV']}/{total_tasks} ({100*mode_totals['UAV']/total_tasks:.2f}%)",
        f"  CLOUD                   : {mode_totals['CLOUD']}/{total_tasks} ({100*mode_totals['CLOUD']/total_tasks:.2f}%)",
        "  缓存更新时隙            : 0/399",
        "  UAV 移动时隙            : 0/399",
        "",
        "已冻结 Proposed（同任务/同外生轨迹）：",
        f"  平均归一化任务代价      : {proposed['mean_task_cost']:.9f}",
        f"  配置更新总成本          : {proposed['config_cost']:.9f}",
        f"  平均 realized 总目标    : {proposed['mean_total_objective']:.9f}",
        "",
        "Static 相对 Proposed：",
        f"  任务代价变化            : {100*task_cost_change:+.2f}%",
        f"  realized 总目标变化     : {100*objective_change:+.2f}%",
        "",
        "说明：",
        "- Static 不运行 P1/P2/Pjoint，也不生成预测 ScenarioSet。",
        "- Static 与 Proposed 使用同一个正式 P3 求解器。",
        "- SV mobility、RSU、Cloud、任务、信道和资源参数均不改变。",
        "- t=0 初始化配置不计更新成本，两种方法口径一致。",
    ]

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig",
    )

    # 保存当前 config 快照，便于确认 baseline 与 Proposed 参数一致。
    shutil.copy2(
        SRC_DIR / "config.py",
        OUT_DIR / "config_snapshot.py",
    )

    manifest_paths = [
        SUMMARY_FILE,
        METRICS_FILE,
        REPORT_FILE,
        OUT_DIR / "config_snapshot.py",
        PROPOSED_SUMMARY,
        SRC_DIR / "p3_solver.py",
        SRC_DIR / "rolling_controller.py",
        SRC_DIR / "cache_model.py",
        SRC_DIR / "uav_state.py",
    ]

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sha256"])
        for path in manifest_paths:
            writer.writerow(
                [
                    str(path.relative_to(PROJECT_ROOT)),
                    sha256_file(path),
                ]
            )

    print()
    print("\n".join(lines))
    print()
    print(f"结果目录                  : {OUT_DIR}")
    print(f"逐时隙结果                : {SUMMARY_FILE}")
    print(f"指标表                    : {METRICS_FILE}")
    print(f"报告                      : {REPORT_FILE}")


if __name__ == "__main__":
    main()
