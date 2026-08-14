#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate the pilot service catalog, dynamic true popularity, and task trace.

Input:
    data/sumo/processed/vehicle_role_pool.csv

Outputs:
    data/workload/service_catalog.csv
    data/workload/popularity_trace.csv
    data/workload/task_trace.csv

This is a PILOT workload for feasibility auditing, not the final frozen
experiment parameter set.

Model choices used here:
- T = 400 slots, delta_t = 1 s
- first 10 TV-ranked vehicles
- F = 10 services
- Bernoulli task arrival per TV per slot, lambda = 0.30
- non-periodic piecewise Zipf popularity with smooth transitions
- task deadline always <= 1 s
- cloud fallback is checked for every generated task
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


T = 400
DELTA_T = 1.0

PILOT_VT = 10
F = 10
ARRIVAL_LAMBDA = 0.30

# Separate reproducibility seeds.
SERVICE_SEED = 101
POPULARITY_SEED = 202
TASK_SEED = 303

# Pilot task ranges.
D_MIN_MBIT = 0.5
D_MAX_MBIT = 1.5
CHI_MIN = 200.0
CHI_MAX = 1000.0
DEADLINE_MIN_S = 0.5
DEADLINE_MAX_S = 1.0

# Service program size range (MB), used later by cache placement.
SERVICE_SIZE_MIN_MB = 0.5
SERVICE_SIZE_MAX_MB = 50.0

# Pilot cloud fallback used only to enforce task legality here.
CLOUD_RATE_BPS = 100e6
CLOUD_CPU_HZ = 20e9
CLOUD_TX_FACTOR = 1.0

# Dynamic popularity.
ZIPF_ALPHAS = (0.6, 0.8, 1.0)
REGIME_MIN_SLOTS = 55
REGIME_MAX_SLOTS = 85
TRANSITION_SLOTS = 10
MIN_FINAL_STABLE_SLOTS = 30


def find_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "data":
        return p.parent.parent
    return Path.cwd()


def numeric_id_key(x: str):
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


def load_tv_ids(role_file: Path, k: int) -> list[str]:
    rows = []
    with role_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["role_pool"] == "TV":
                rows.append((int(row["role_rank"]), row["vehicle_id"]))
    rows.sort()
    ids = [vehicle_id for _, vehicle_id in rows[:k]]
    if len(ids) != k:
        raise RuntimeError(f"Need {k} TV candidates, found {len(ids)}")
    return ids


def make_service_catalog():
    rng = random.Random(SERVICE_SEED)

    # Heterogeneous service means. Lists are shuffled independently so
    # "large input", "high compute", and "tight deadline" do not always
    # describe the same service.
    d_means = [
        D_MIN_MBIT + (D_MAX_MBIT - D_MIN_MBIT) * i / (F - 1)
        for i in range(F)
    ]
    chi_means = [
        300.0 + (900.0 - 300.0) * i / (F - 1)
        for i in range(F)
    ]
    deadlines = [
        0.58 + (0.95 - 0.58) * i / (F - 1)
        for i in range(F)
    ]

    rng.shuffle(d_means)
    rng.shuffle(chi_means)
    rng.shuffle(deadlines)

    catalog = []
    for f in range(F):
        catalog.append(
            {
                "service_id": f,
                "program_size_MB": rng.uniform(
                    SERVICE_SIZE_MIN_MB, SERVICE_SIZE_MAX_MB
                ),
                "D_mean_Mbit": d_means[f],
                "chi_mean_cycles_per_bit": chi_means[f],
                "deadline_mean_s": deadlines[f],
            }
        )
    return catalog


def zipf_probs(ranking: list[int], alpha: float) -> list[float]:
    weights_by_service = [0.0] * F
    denom = sum((r + 1) ** (-alpha) for r in range(F))
    for rank, service_id in enumerate(ranking):
        weights_by_service[service_id] = (
            (rank + 1) ** (-alpha) / denom
        )
    return weights_by_service


def changed_ranking(
    old: list[int],
    rng: random.Random,
    severity: str,
) -> list[int]:
    new = old[:]

    if severity == "small":
        i, j = rng.sample(range(F), 2)
        new[i], new[j] = new[j], new[i]
    elif severity == "medium":
        for _ in range(3):
            i, j = rng.sample(range(F), 2)
            new[i], new[j] = new[j], new[i]
    elif severity == "large":
        rng.shuffle(new)
    else:
        raise ValueError(severity)

    if new == old:
        # Defensive only; normally impossible except an accidental full shuffle.
        new = old[1:] + old[:1]

    return new


def make_popularity_trace():
    rng = random.Random(POPULARITY_SEED)

    ranking = list(range(F))
    rng.shuffle(ranking)
    alpha = rng.choice(ZIPF_ALPHAS)

    current_probs = zipf_probs(ranking, alpha)
    trace = []
    regime_id = 0
    slot = 0
    change_slots = []

    while slot < T:
        duration = rng.randint(REGIME_MIN_SLOTS, REGIME_MAX_SLOTS)
        stable_end = min(T, slot + duration)

        # Stable portion.
        while slot < stable_end:
            trace.append(
                {
                    "slot": slot,
                    "regime_id": regime_id,
                    "phase": "stable",
                    "alpha": alpha,
                    "severity": "none",
                    "gamma": 0.0,
                    "probs": current_probs[:],
                }
            )
            slot += 1

        if slot >= T:
            break

        if T - slot < TRANSITION_SLOTS + MIN_FINAL_STABLE_SLOTS:
            while slot < T:
                trace.append(
                    {
                        "slot": slot,
                        "regime_id": regime_id,
                        "phase": "stable",
                        "alpha": alpha,
                        "severity": "none",
                        "gamma": 0.0,
                        "probs": current_probs[:],
                    }
                )
                slot += 1
            break

        severity = rng.choice(["small", "medium", "large"])
        new_ranking = changed_ranking(ranking, rng, severity)

        alpha_candidates = [a for a in ZIPF_ALPHAS if a != alpha]
        new_alpha = rng.choice(alpha_candidates)
        new_probs = zipf_probs(new_ranking, new_alpha)

        change_slots.append(slot)
        transition_len = TRANSITION_SLOTS

        for j in range(transition_len):
            gamma = (j + 1) / transition_len
            mixed = [
                (1.0 - gamma) * p_old + gamma * p_new
                for p_old, p_new in zip(current_probs, new_probs)
            ]
            trace.append(
                {
                    "slot": slot,
                    "regime_id": regime_id + 1,
                    "phase": "transition",
                    "alpha": new_alpha,
                    "severity": severity,
                    "gamma": gamma,
                    "probs": mixed,
                }
            )
            slot += 1

        ranking = new_ranking
        alpha = new_alpha
        current_probs = new_probs
        regime_id += 1

    if len(trace) != T:
        raise RuntimeError(f"Popularity length {len(trace)} != T={T}")

    return trace, change_slots


def sample_task_for_service(service, rng):
    d_mean = service["D_mean_Mbit"]
    chi_mean = service["chi_mean_cycles_per_bit"]
    deadline_mean = service["deadline_mean_s"]

    for _ in range(100):
        d_mbit = min(
            D_MAX_MBIT,
            max(D_MIN_MBIT, rng.uniform(0.8, 1.2) * d_mean),
        )
        chi = min(
            CHI_MAX,
            max(CHI_MIN, rng.uniform(0.9, 1.1) * chi_mean),
        )
        deadline = min(
            DEADLINE_MAX_S,
            max(
                DEADLINE_MIN_S,
                deadline_mean + rng.uniform(-0.05, 0.05),
            ),
        )

        d_bits = d_mbit * 1e6
        l_cycles = d_bits * chi

        cloud_delay = (
            CLOUD_TX_FACTOR * d_bits / CLOUD_RATE_BPS
            + l_cycles / CLOUD_CPU_HZ
        )

        if cloud_delay <= deadline <= DELTA_T:
            return d_bits, chi, l_cycles, deadline, cloud_delay

    raise RuntimeError("Failed to sample a cloud-feasible legal task.")


def choose_service(probs: list[float], rng: random.Random) -> int:
    return rng.choices(range(F), weights=probs, k=1)[0]


def write_outputs(root: Path):
    role_file = (
        root / "data" / "sumo" / "processed" / "vehicle_role_pool.csv"
    )
    if not role_file.is_file():
        raise FileNotFoundError(role_file)

    tv_ids = load_tv_ids(role_file, PILOT_VT)
    catalog = make_service_catalog()
    popularity, change_slots = make_popularity_trace()

    out_dir = root / "data" / "workload"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_file = out_dir / "service_catalog.csv"
    pop_file = out_dir / "popularity_trace.csv"
    task_file = out_dir / "task_trace.csv"

    with catalog_file.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "service_id",
            "program_size_MB",
            "D_mean_Mbit",
            "chi_mean_cycles_per_bit",
            "deadline_mean_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in catalog:
            w.writerow(
                {
                    "service_id": row["service_id"],
                    "program_size_MB": f"{row['program_size_MB']:.6f}",
                    "D_mean_Mbit": f"{row['D_mean_Mbit']:.6f}",
                    "chi_mean_cycles_per_bit":
                        f"{row['chi_mean_cycles_per_bit']:.6f}",
                    "deadline_mean_s": f"{row['deadline_mean_s']:.6f}",
                }
            )

    with pop_file.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "slot",
            "regime_id",
            "phase",
            "alpha",
            "severity",
            "gamma",
        ] + [f"rho_{i}" for i in range(F)]

        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in popularity:
            out = {
                "slot": row["slot"],
                "regime_id": row["regime_id"],
                "phase": row["phase"],
                "alpha": row["alpha"],
                "severity": row["severity"],
                "gamma": f"{row['gamma']:.6f}",
            }
            for i, p in enumerate(row["probs"]):
                out[f"rho_{i}"] = f"{p:.10f}"
            w.writerow(out)

    rng = random.Random(TASK_SEED)
    service_counts = [0] * F
    slot_counts = [0] * T
    task_count = 0
    min_cloud_margin = math.inf

    with task_file.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "slot",
            "vehicle_id",
            "service_id",
            "D_bits",
            "cycles_per_bit",
            "L_cycles",
            "deadline_s",
            "cloud_delay_s",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for slot in range(T):
            probs = popularity[slot]["probs"]

            for vehicle_id in tv_ids:
                if rng.random() >= ARRIVAL_LAMBDA:
                    continue

                service_id = choose_service(probs, rng)
                service = catalog[service_id]

                d_bits, chi, l_cycles, deadline, cloud_delay = (
                    sample_task_for_service(service, rng)
                )

                w.writerow(
                    {
                        "task_id": task_count,
                        "slot": slot,
                        "vehicle_id": vehicle_id,
                        "service_id": service_id,
                        "D_bits": f"{d_bits:.3f}",
                        "cycles_per_bit": f"{chi:.6f}",
                        "L_cycles": f"{l_cycles:.3f}",
                        "deadline_s": f"{deadline:.6f}",
                        "cloud_delay_s": f"{cloud_delay:.6f}",
                    }
                )

                task_count += 1
                service_counts[service_id] += 1
                slot_counts[slot] += 1
                min_cloud_margin = min(
                    min_cloud_margin, deadline - cloud_delay
                )

    empirical_lambda = task_count / (PILOT_VT * T)

    print("=" * 72)
    print("Pilot workload generated")
    print("=" * 72)
    print(f"T                    : {T}")
    print(f"TVs                  : {PILOT_VT}")
    print(f"Services F           : {F}")
    print(f"Target arrival lambda: {ARRIVAL_LAMBDA:.3f}")
    print(f"Empirical lambda     : {empirical_lambda:.4f}")
    print(f"Total tasks          : {task_count}")
    print(f"Tasks/slot mean      : {task_count / T:.3f}")
    print(f"Tasks/slot max       : {max(slot_counts)}")
    print(f"Popularity changes   : {len(change_slots)}")
    print(f"Change slots         : {change_slots}")
    print(f"Min cloud margin     : {min_cloud_margin:.6f} s")
    print(f"TV IDs               : {', '.join(tv_ids)}")
    print()
    print("Service request counts:")
    for i, n in enumerate(service_counts):
        print(f"  service {i}: {n}")
    print()
    print("Files:")
    print(f"  {catalog_file}")
    print(f"  {pop_file}")
    print(f"  {task_file}")


if __name__ == "__main__":
    write_outputs(find_root())
