#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit TV -> SV full-resource V2V link feasibility using the actual
400-slot SUMO mobility episode.

Inputs:
    data/sumo/processed/mobility_episode_400.csv
    data/sumo/processed/vehicle_role_pool.csv
    data/workload/task_trace.csv

No output files are written.

Pilot radio assumptions:
- first 5 ranked SVs
- 5.9 GHz V2V mean path-loss model for urban low-density measurements:
      PL(d)[dB] = 59.37 + 10 * 1.75 * log10(d[m])
  Source:
      H. Fernandez et al.,
      "Path Loss Characterization for Vehicular Communications at
       700 MHz and 5.9 GHz Under LOS and NLOS Conditions,"
      IEEE Antennas and Wireless Propagation Letters, 2014,
      DOI: 10.1109/LAWP.2014.2322261
- B = 10 MHz
- transmit power = 0.1 W
- thermal noise PSD = -174 dBm/Hz
- receiver noise figure = 9 dB (pilot engineering assumption)
- SV CPU = 3 GHz
- no random shadowing in this first geometry audit

The measurement paper reports U-LD sampled separation spans roughly
8--216 m across LOS/NLOS routes. Rates beyond this range are printed
as an extrapolation diagnostic, not treated as empirical validation.
"""

import csv
import math
from pathlib import Path


PILOT_SV = 5
SV_CPU_HZ = 3e9

B_HZ = 10e6
TX_POWER_W = 0.1
NOISE_PSD_DBM_HZ = -174.0
NOISE_FIGURE_DB = 9.0

PL0_DB = 59.37
PATHLOSS_EXPONENT = 1.75

MEASURED_D_MIN_M = 8.0
MEASURED_D_MAX_M = 216.0


def find_root():
    p = Path(__file__).resolve()
    if p.parent.name == "data":
        return p.parent.parent
    return Path.cwd()


def percentile(values, q):
    x = sorted(values)
    if not x:
        return float("nan")
    pos = (len(x) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(x) - 1)
    w = pos - lo
    return x[lo] * (1.0 - w) + x[hi] * w


def summary(name, values, unit=""):
    print(
        f"{name:<30}: "
        f"P05={percentile(values, 0.05):.3f}, "
        f"P50={percentile(values, 0.50):.3f}, "
        f"P95={percentile(values, 0.95):.3f} {unit}"
    )


def dbm_to_w(dbm):
    return 10 ** ((dbm - 30.0) / 10.0)


def full_rate_bps(distance_m, noise_w):
    # Avoid singular behavior below the lower measured separation.
    d = max(distance_m, MEASURED_D_MIN_M)
    pl_db = PL0_DB + 10.0 * PATHLOSS_EXPONENT * math.log10(d)
    gain = 10 ** (-pl_db / 10.0)
    snr = TX_POWER_W * gain / noise_w
    return B_HZ * math.log2(1.0 + snr)


def load_sv_ids(role_file):
    rows = []
    with role_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["role_pool"] == "SV":
                rows.append((int(row["role_rank"]), row["vehicle_id"]))
    rows.sort()
    ids = [vid for _, vid in rows[:PILOT_SV]]
    if len(ids) != PILOT_SV:
        raise RuntimeError(f"Need {PILOT_SV} SVs, found {len(ids)}")
    return ids


def main():
    root = find_root()
    mobility_file = root / "data" / "sumo" / "processed" / "mobility_episode_400.csv"
    role_file = root / "data" / "sumo" / "processed" / "vehicle_role_pool.csv"
    task_file = root / "data" / "workload" / "task_trace.csv"

    for p in (mobility_file, role_file, task_file):
        if not p.is_file():
            raise FileNotFoundError(p)

    sv_ids = load_sv_ids(role_file)
    sv_set = set(sv_ids)

    # positions[(slot, vehicle_id)] = (x, y)
    positions = {}
    with mobility_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vid = row["vehicle_id"]
            if vid in sv_set:
                positions[(int(row["slot"]), vid)] = (
                    float(row["x"]),
                    float(row["y"]),
                )
            else:
                # TV positions are also needed for actual tasks.
                positions[(int(row["slot"]), vid)] = (
                    float(row["x"]),
                    float(row["y"]),
                )

    tasks = []
    with task_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tasks.append(
                {
                    "slot": int(row["slot"]),
                    "vehicle_id": row["vehicle_id"],
                    "D": float(row["D_bits"]),
                    "L": float(row["L_cycles"]),
                    "deadline": float(row["deadline_s"]),
                }
            )

    noise_dbm = (
        NOISE_PSD_DBM_HZ
        + 10.0 * math.log10(B_HZ)
        + NOISE_FIGURE_DB
    )
    noise_w = dbm_to_w(noise_dbm)

    nearest_distances = []
    best_rates_mbps = []
    required_rates_mbps = []
    candidate_counts = []

    any_full_resource_ok = 0
    any_measured_range = 0
    any_measured_range_and_ok = 0
    link_count = 0
    link_in_measured_range = 0

    for task in tasks:
        key_tv = (task["slot"], task["vehicle_id"])
        if key_tv not in positions:
            raise RuntimeError(f"Missing TV position: {key_tv}")

        tx, ty = positions[key_tv]

        compute_delay = task["L"] / SV_CPU_HZ
        remaining = task["deadline"] - compute_delay
        if remaining <= 0:
            required_rate = math.inf
        else:
            required_rate = task["D"] / remaining

        required_rates_mbps.append(required_rate / 1e6)

        distances = []
        rates = []
        ok_count = 0
        measured_range_seen = False
        measured_range_ok = False

        for sv in sv_ids:
            key_sv = (task["slot"], sv)
            if key_sv not in positions:
                raise RuntimeError(f"Missing SV position: {key_sv}")

            sx, sy = positions[key_sv]
            d = math.hypot(tx - sx, ty - sy)
            r = full_rate_bps(d, noise_w)

            distances.append(d)
            rates.append(r)
            link_count += 1

            if MEASURED_D_MIN_M <= d <= MEASURED_D_MAX_M:
                link_in_measured_range += 1
                measured_range_seen = True
                if r >= required_rate:
                    measured_range_ok = True

            if r >= required_rate:
                ok_count += 1

        nearest_distances.append(min(distances))
        best_rates_mbps.append(max(rates) / 1e6)
        candidate_counts.append(ok_count)

        if ok_count > 0:
            any_full_resource_ok += 1
        if measured_range_seen:
            any_measured_range += 1
        if measured_range_ok:
            any_measured_range_and_ok += 1

    n = len(tasks)

    print("=" * 76)
    print("TV -> SV V2V channel audit")
    print("=" * 76)
    print(f"Tasks                       : {n}")
    print(f"Pilot SVs                   : {PILOT_SV}")
    print(f"SV IDs                      : {', '.join(sv_ids)}")
    print(f"Bandwidth                   : {B_HZ/1e6:.1f} MHz")
    print(f"TX power                    : {TX_POWER_W:.3f} W")
    print(f"Noise power                 : {noise_dbm:.2f} dBm")
    print(f"Path loss                   : 59.37 + 17.5 log10(d[m]) dB")
    print()

    summary("Nearest SV distance", nearest_distances, "m")
    summary("Required SV uplink rate", required_rates_mbps, "Mbps")
    summary("Best full-resource rate", best_rates_mbps, "Mbps")
    summary("Feasible SV count/task", candidate_counts, "")
    print()

    print(
        f"Task has >=1 SV full-rate OK : "
        f"{any_full_resource_ok}/{n} "
        f"({100.0 * any_full_resource_ok / n:.2f}%)"
    )
    print(
        f"Task has SV in 8..216 m      : "
        f"{any_measured_range}/{n} "
        f"({100.0 * any_measured_range / n:.2f}%)"
    )
    print(
        f"In-range SV and deadline OK   : "
        f"{any_measured_range_and_ok}/{n} "
        f"({100.0 * any_measured_range_and_ok / n:.2f}%)"
    )
    print(
        f"All task-SV links in 8..216 m : "
        f"{link_in_measured_range}/{link_count} "
        f"({100.0 * link_in_measured_range / link_count:.2f}%)"
    )
    print()
    print("Important:")
    print("- Full-resource rate is only a P3 eligibility upper bound.")
    print("- Actual P3 rate is b * R_bar with shared b, so congestion still matters.")
    print("- Values beyond 216 m use model extrapolation and are diagnostic only.")
    print("- Do not freeze radio parameters until this output is inspected.")


if __name__ == "__main__":
    main()
