#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path


TV_CPU_HZ = 1e9
SV_CPU_HZ = 3e9
RSU_CPU_HZ = 10e9
UAV_CPU_HZ = 10e9

TASK_FILE = Path("data/workload/task_trace.csv")


def percentile(values, q):
    x = sorted(values)
    if not x:
        return float("nan")
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    w = pos - lo
    return x[lo] * (1 - w) + x[hi] * w


def summarize(name, values, scale=1.0, unit=""):
    vals = [v * scale for v in values]
    print(
        f"{name:<24}: "
        f"P05={percentile(vals, 0.05):.3f}, "
        f"P50={percentile(vals, 0.50):.3f}, "
        f"P95={percentile(vals, 0.95):.3f} {unit}"
    )


def main():
    if not TASK_FILE.is_file():
        raise FileNotFoundError(TASK_FILE)

    tasks = []
    with TASK_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tasks.append(
                {
                    "D": float(row["D_bits"]),
                    "L": float(row["L_cycles"]),
                    "deadline": float(row["deadline_s"]),
                    "cloud_delay": float(row["cloud_delay_s"]),
                }
            )

    if not tasks:
        raise RuntimeError("No tasks found.")

    local_delay = [x["L"] / TV_CPU_HZ for x in tasks]
    cloud_delay = [x["cloud_delay"] for x in tasks]
    required_cpu = [x["L"] / x["deadline"] for x in tasks]

    local_ok = [
        d <= x["deadline"] for d, x in zip(local_delay, tasks)
    ]
    cloud_ok = [
        d <= x["deadline"] for d, x in zip(cloud_delay, tasks)
    ]

    print("=" * 72)
    print("Task-side feasibility audit")
    print("=" * 72)
    print(f"Tasks                  : {len(tasks)}")
    print(f"TV CPU                 : {TV_CPU_HZ/1e9:.1f} GHz")
    print(f"SV CPU                 : {SV_CPU_HZ/1e9:.1f} GHz")
    print(f"RSU/UAV CPU            : {RSU_CPU_HZ/1e9:.1f} GHz")
    print()

    summarize(
        "Input size D",
        [x["D"] for x in tasks],
        scale=1e-6,
        unit="Mbit",
    )
    summarize(
        "Cycles L",
        [x["L"] for x in tasks],
        scale=1e-9,
        unit="Gcycles",
    )
    summarize(
        "Deadline",
        [x["deadline"] for x in tasks],
        unit="s",
    )
    summarize(
        "Required CPU L/Tmax",
        required_cpu,
        scale=1e-9,
        unit="GHz",
    )
    print()

    print(
        f"Local feasible          : "
        f"{sum(local_ok)}/{len(tasks)} "
        f"({100*sum(local_ok)/len(tasks):.2f}%)"
    )
    print(
        f"Cloud feasible          : "
        f"{sum(cloud_ok)}/{len(tasks)} "
        f"({100*sum(cloud_ok)/len(tasks):.2f}%)"
    )
    summarize("Local delay", local_delay, unit="s")
    summarize("Cloud delay", cloud_delay, unit="s")
    print()

    # For an edge node using full CPU share w=1, the remaining deadline
    # determines the minimum uplink rate needed:
    #   D/R + L/F <= Tmax  =>  R >= D/(Tmax - L/F)
    for node_name, cpu in [
        ("SV", SV_CPU_HZ),
        ("RSU/UAV", RSU_CPU_HZ),
    ]:
        compute_only_ok = 0
        required_rates = []

        for x in tasks:
            compute_delay = x["L"] / cpu
            remaining = x["deadline"] - compute_delay

            if remaining > 0:
                compute_only_ok += 1
                required_rates.append(x["D"] / remaining)

        print(
            f"{node_name} compute-only OK    : "
            f"{compute_only_ok}/{len(tasks)} "
            f"({100*compute_only_ok/len(tasks):.2f}%)"
        )

        if required_rates:
            summarize(
                f"{node_name} min uplink rate",
                required_rates,
                scale=1e-6,
                unit="Mbps",
            )
        print()

    print("Interpretation:")
    print("- Local feasibility should be neither ~0% nor ~100% for a useful pilot.")
    print("- Cloud being 100% feasible is allowed as fallback, but it must not")
    print("  automatically dominate utility once transmission energy/cost is added.")
    print("- Next step: compare the required edge uplink-rate percentiles above")
    print("  with rates produced by the actual mobility + channel model.")


if __name__ == "__main__":
    main()
