#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pilot P3 solver: SV + Cloud only, no local execution.

Inputs:
    data/sumo/processed/mobility_episode_400.csv
    data/sumo/processed/vehicle_role_pool.csv
    data/workload/task_trace.csv

Output:
    results/p3_sv_cloud_pilot.csv

Purpose:
    Validate the P3 offloading/resource-allocation core before adding
    RSUs, UAVs, and optimized cache states.

Pilot assumptions that are NOT frozen paper parameters:
    - first 5 ranked SVs
    - every pilot SV can serve every service (cache gate temporarily open)
    - omega_T = omega_E = 0.5
    - cloud: R_c = 100 Mbps, F_c = 20 GHz, psi_c = 1
    - V2V: B = 10 MHz, p = 0.1 W, noise figure = 9 dB
    - V2V candidate only when 8 m <= effective distance <= 216 m
      (actual distance must be <= 216 m; distances below 8 m use d=8 m
       only inside the path-loss formula)
    - mean V2V path loss:
          PL(d) = 59.37 + 17.5 log10(d[m]) dB

Model:
    Edge delay:
        T = alpha / w + beta / b
        alpha = L / F_s
        beta  = D / R_bar

    With E_ref = p * T_max and omega_T + omega_E = 1:
        J_edge = A / w + Bcoef / b
        A      = omega_T * L / (T_max * F_s)
        Bcoef  = D / (R_bar * T_max)

    Cloud:
        T_cloud = psi_c * D/R_c + L/F_c
        J_cloud = omega_T*T_cloud/T_max
                  + omega_E*(D/R_c)/T_max
"""

from __future__ import annotations

import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Pilot configuration
# ---------------------------------------------------------------------------

PILOT_SV = 5

OMEGA_T = 0.5
OMEGA_E = 0.5

SV_CPU_HZ = 3e9

CLOUD_RATE_BPS = 100e6
CLOUD_CPU_HZ = 20e9
CLOUD_PSI = 1.0

B_HZ = 10e6
TX_POWER_W = 0.1
NOISE_PSD_DBM_HZ = -174.0
NOISE_FIGURE_DB = 9.0

PL0_DB = 59.37
PATHLOSS_EXPONENT = 1.75
PATHLOSS_D_FLOOR_M = 8.0
V2V_MAX_DISTANCE_M = 216.0

EPS = 1e-8
FEAS_TOL = 1e-6


def find_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name in {"src", "data"}:
        return p.parent.parent
    return Path.cwd()


def dbm_to_w(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


NOISE_DBM = (
    NOISE_PSD_DBM_HZ
    + 10.0 * math.log10(B_HZ)
    + NOISE_FIGURE_DB
)
NOISE_W = dbm_to_w(NOISE_DBM)


def full_rate_bps(distance_m: float) -> float:
    d_eff = max(distance_m, PATHLOSS_D_FLOOR_M)
    pl_db = PL0_DB + 10.0 * PATHLOSS_EXPONENT * math.log10(d_eff)
    gain = 10 ** (-pl_db / 10.0)
    snr = TX_POWER_W * gain / NOISE_W
    return B_HZ * math.log2(1.0 + snr)


def is_service_cached(sv_id: str, service_id: int) -> bool:
    """
    Pilot cache gate.

    P3 already calls this function for every edge candidate. For this first
    solver validation all services are available. Later replace this body
    with lookup of the actual c_{s,f} cache state; the rest of P3 need not
    change.
    """
    return True


def load_sv_ids(role_file: Path) -> list[str]:
    ranked = []
    with role_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["role_pool"] == "SV":
                ranked.append((int(row["role_rank"]), row["vehicle_id"]))
    ranked.sort()

    ids = [vid for _, vid in ranked[:PILOT_SV]]
    if len(ids) != PILOT_SV:
        raise RuntimeError(f"Need {PILOT_SV} SVs, found {len(ids)}")
    return ids


def load_positions(mobility_file: Path) -> dict[tuple[int, str], tuple[float, float]]:
    positions = {}
    with mobility_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            positions[(int(row["slot"]), row["vehicle_id"])] = (
                float(row["x"]),
                float(row["y"]),
            )
    return positions


def load_tasks(task_file: Path) -> list[dict]:
    tasks = []
    with task_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tasks.append(
                {
                    "task_id": int(row["task_id"]),
                    "slot": int(row["slot"]),
                    "vehicle_id": row["vehicle_id"],
                    "service_id": int(row["service_id"]),
                    "D": float(row["D_bits"]),
                    "L": float(row["L_cycles"]),
                    "Tmax": float(row["deadline_s"]),
                }
            )
    return tasks


def cloud_metrics(task: dict) -> dict:
    tx_time = CLOUD_PSI * task["D"] / CLOUD_RATE_BPS
    compute_time = task["L"] / CLOUD_CPU_HZ
    delay = tx_time + compute_time

    # E_ref = p*Tmax, hence p cancels in the normalized cloud energy term.
    energy_ratio = (task["D"] / CLOUD_RATE_BPS) / task["Tmax"]
    cost = (
        OMEGA_T * delay / task["Tmax"]
        + OMEGA_E * energy_ratio
    )

    return {
        "delay": delay,
        "energy_J": TX_POWER_W * task["D"] / CLOUD_RATE_BPS,
        "cost": cost,
    }


def make_edge_candidate(
    task: dict,
    sv_id: str,
    positions: dict,
) -> dict | None:
    if not is_service_cached(sv_id, task["service_id"]):
        return None

    key_tv = (task["slot"], task["vehicle_id"])
    key_sv = (task["slot"], sv_id)

    if key_tv not in positions:
        raise RuntimeError(f"Missing TV position {key_tv}")
    if key_sv not in positions:
        raise RuntimeError(f"Missing SV position {key_sv}")

    tx, ty = positions[key_tv]
    sx, sy = positions[key_sv]
    distance = math.hypot(tx - sx, ty - sy)

    # Do not extrapolate the measurement-based V2V model beyond 216 m.
    if distance > V2V_MAX_DISTANCE_M:
        return None

    rbar = full_rate_bps(distance)

    alpha = task["L"] / SV_CPU_HZ
    beta = task["D"] / rbar

    # Even with full CPU and bandwidth this SV cannot meet the deadline.
    if alpha + beta > task["Tmax"] + FEAS_TOL:
        return None

    A = OMEGA_T * task["L"] / (task["Tmax"] * SV_CPU_HZ)

    # Since E_ref = p*Tmax and omega_T + omega_E = 1:
    Bcoef = task["D"] / (rbar * task["Tmax"])

    return {
        "task_id": task["task_id"],
        "sv_id": sv_id,
        "distance_m": distance,
        "rbar_bps": rbar,
        "alpha": alpha,
        "beta": beta,
        "A": A,
        "Bcoef": Bcoef,
        "Tmax": task["Tmax"],
    }


def closed_form_allocation(cands: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    sqrtA = np.sqrt(np.array([c["A"] for c in cands], dtype=float))
    sqrtB = np.sqrt(np.array([c["Bcoef"] for c in cands], dtype=float))

    w = sqrtA / sqrtA.sum()
    b = sqrtB / sqrtB.sum()
    return w, b


def allocation_is_feasible(
    cands: list[dict],
    w: np.ndarray,
    b: np.ndarray,
) -> bool:
    if np.any(w <= 0.0) or np.any(b <= 0.0):
        return False
    if w.sum() > 1.0 + FEAS_TOL or b.sum() > 1.0 + FEAS_TOL:
        return False

    for i, c in enumerate(cands):
        delay = c["alpha"] / w[i] + c["beta"] / b[i]
        if delay > c["Tmax"] + FEAS_TOL:
            return False

    return True


def solve_node_allocation(cands: list[dict]) -> dict | None:
    """
    Solve one SV's convex resource-allocation problem for a fixed task set.

    Returns None if the assignment is resource-infeasible.
    """
    n = len(cands)
    if n == 0:
        return {"cost": 0.0, "alloc": {}, "used_slsqp": False}

    w0, b0 = closed_form_allocation(cands)

    if allocation_is_feasible(cands, w0, b0):
        w = w0
        b = b0
        used_slsqp = False
    else:
        A = np.array([c["A"] for c in cands], dtype=float)
        Bcoef = np.array([c["Bcoef"] for c in cands], dtype=float)
        alpha = np.array([c["alpha"] for c in cands], dtype=float)
        beta = np.array([c["beta"] for c in cands], dtype=float)
        Tmax = np.array([c["Tmax"] for c in cands], dtype=float)

        def objective(x):
            wv = x[:n]
            bv = x[n:]
            return float(np.sum(A / wv + Bcoef / bv))

        def resource_constraints(x):
            wv = x[:n]
            bv = x[n:]
            return np.array([
                1.0 - np.sum(wv),
                1.0 - np.sum(bv),
            ])

        def deadline_constraints(x):
            wv = x[:n]
            bv = x[n:]
            return Tmax - alpha / wv - beta / bv

        x0 = np.concatenate([w0, b0])

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(EPS, 1.0)] * (2 * n),
            constraints=[
                {"type": "ineq", "fun": resource_constraints},
                {"type": "ineq", "fun": deadline_constraints},
            ],
            options={
                "ftol": 1e-10,
                "maxiter": 300,
                "disp": False,
            },
        )

        if not res.success:
            return None

        w = res.x[:n]
        b = res.x[n:]

        if not allocation_is_feasible(cands, w, b):
            return None

        used_slsqp = True

    alloc = {}
    total_cost = 0.0

    for i, c in enumerate(cands):
        delay = c["alpha"] / w[i] + c["beta"] / b[i]
        energy_J = TX_POWER_W * c["beta"] / b[i]
        cost = c["A"] / w[i] + c["Bcoef"] / b[i]

        total_cost += cost
        alloc[c["task_id"]] = {
            "w": float(w[i]),
            "b": float(b[i]),
            "delay": float(delay),
            "energy_J": float(energy_J),
            "cost": float(cost),
            "distance_m": c["distance_m"],
            "rbar_bps": c["rbar_bps"],
        }

    return {
        "cost": float(total_cost),
        "alloc": alloc,
        "used_slsqp": used_slsqp,
    }


def solve_slot(
    slot_tasks: list[dict],
    sv_ids: list[str],
    positions: dict,
) -> tuple[dict, dict]:
    """
    Exact enumeration over Cloud / eligible SV choices for this slot.
    For every assignment, each SV resource subproblem is solved independently.
    """
    task_by_id = {t["task_id"]: t for t in slot_tasks}

    cloud = {}
    edge_candidates = {}
    option_lists = []

    eligible_task_count = 0

    for task in slot_tasks:
        cm = cloud_metrics(task)
        if cm["delay"] > task["Tmax"] + FEAS_TOL:
            raise RuntimeError(
                f"Cloud fallback infeasible for task {task['task_id']}"
            )
        cloud[task["task_id"]] = cm

        cand_map = {}
        for sv_id in sv_ids:
            c = make_edge_candidate(task, sv_id, positions)
            if c is not None:
                cand_map[sv_id] = c

        edge_candidates[task["task_id"]] = cand_map

        if cand_map:
            eligible_task_count += 1

        # Cloud is always an option.
        option_lists.append(["cloud"] + list(cand_map.keys()))

    best_cost = math.inf
    best_result = None

    assignments_evaluated = 0
    assignments_feasible = 0
    slsqp_calls_in_best = 0

    for choices in itertools.product(*option_lists):
        assignments_evaluated += 1

        total_cost = 0.0
        result = {}
        groups = defaultdict(list)
        feasible = True
        used_slsqp_count = 0

        for task, choice in zip(slot_tasks, choices):
            tid = task["task_id"]

            if choice == "cloud":
                cm = cloud[tid]
                total_cost += cm["cost"]
                result[tid] = {
                    "mode": "cloud",
                    "node_id": "cloud",
                    "w": None,
                    "b": None,
                    "delay": cm["delay"],
                    "energy_J": cm["energy_J"],
                    "cost": cm["cost"],
                    "distance_m": None,
                    "rbar_bps": CLOUD_RATE_BPS,
                }
            else:
                groups[choice].append(edge_candidates[tid][choice])

        for sv_id, cands in groups.items():
            solved = solve_node_allocation(cands)
            if solved is None:
                feasible = False
                break

            total_cost += solved["cost"]
            used_slsqp_count += int(solved["used_slsqp"])

            for tid, a in solved["alloc"].items():
                result[tid] = {
                    "mode": "sv",
                    "node_id": sv_id,
                    **a,
                }

        if not feasible:
            continue

        assignments_feasible += 1

        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            best_result = result
            slsqp_calls_in_best = used_slsqp_count

    if best_result is None:
        raise RuntimeError(
            f"No feasible P3 assignment for slot {slot_tasks[0]['slot']}"
        )

    return best_result, {
        "eligible_tasks": eligible_task_count,
        "assignments_evaluated": assignments_evaluated,
        "assignments_feasible": assignments_feasible,
        "best_slsqp_nodes": slsqp_calls_in_best,
    }


def main():
    if abs(OMEGA_T + OMEGA_E - 1.0) > 1e-12:
        raise ValueError("OMEGA_T + OMEGA_E must equal 1.")

    root = find_root()

    mobility_file = (
        root / "data" / "sumo" / "processed" / "mobility_episode_400.csv"
    )
    role_file = (
        root / "data" / "sumo" / "processed" / "vehicle_role_pool.csv"
    )
    task_file = root / "data" / "workload" / "task_trace.csv"
    out_file = root / "results" / "p3_sv_cloud_pilot.csv"

    for p in (mobility_file, role_file, task_file):
        if not p.is_file():
            raise FileNotFoundError(p)

    sv_ids = load_sv_ids(role_file)
    positions = load_positions(mobility_file)
    tasks = load_tasks(task_file)

    tasks_by_slot = defaultdict(list)
    for task in tasks:
        tasks_by_slot[task["slot"]].append(task)

    # Useful baseline: objective if every task goes to cloud.
    all_cloud_cost = sum(cloud_metrics(t)["cost"] for t in tasks)

    all_results = {}
    total_assignments = 0
    total_feasible_assignments = 0
    slots_with_sv_eligible = 0
    eligible_tasks = 0
    best_slsqp_nodes = 0

    for slot in sorted(tasks_by_slot):
        result, stats = solve_slot(
            tasks_by_slot[slot],
            sv_ids,
            positions,
        )
        all_results.update(result)

        total_assignments += stats["assignments_evaluated"]
        total_feasible_assignments += stats["assignments_feasible"]
        eligible_tasks += stats["eligible_tasks"]
        best_slsqp_nodes += stats["best_slsqp_nodes"]

        if stats["eligible_tasks"] > 0:
            slots_with_sv_eligible += 1

    out_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "task_id",
        "slot",
        "vehicle_id",
        "service_id",
        "mode",
        "node_id",
        "delay_s",
        "deadline_s",
        "energy_J",
        "cost",
        "w_cpu",
        "b_bw",
        "distance_m",
        "full_rate_Mbps",
    ]

    mode_counts = Counter()
    node_counts = Counter()
    total_cost = 0.0
    total_delay = 0.0
    total_energy = 0.0
    deadline_violations = 0

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for task in sorted(tasks, key=lambda x: x["task_id"]):
            r = all_results[task["task_id"]]

            mode_counts[r["mode"]] += 1
            node_counts[r["node_id"]] += 1
            total_cost += r["cost"]
            total_delay += r["delay"]
            total_energy += r["energy_J"]

            if r["delay"] > task["Tmax"] + FEAS_TOL:
                deadline_violations += 1

            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "slot": task["slot"],
                    "vehicle_id": task["vehicle_id"],
                    "service_id": task["service_id"],
                    "mode": r["mode"],
                    "node_id": r["node_id"],
                    "delay_s": f"{r['delay']:.9f}",
                    "deadline_s": f"{task['Tmax']:.9f}",
                    "energy_J": f"{r['energy_J']:.12f}",
                    "cost": f"{r['cost']:.12f}",
                    "w_cpu": "" if r["w"] is None else f"{r['w']:.9f}",
                    "b_bw": "" if r["b"] is None else f"{r['b']:.9f}",
                    "distance_m": (
                        ""
                        if r["distance_m"] is None
                        else f"{r['distance_m']:.6f}"
                    ),
                    "full_rate_Mbps": f"{r['rbar_bps']/1e6:.6f}",
                }
            )

    n = len(tasks)
    improvement = (
        100.0 * (all_cloud_cost - total_cost) / all_cloud_cost
        if all_cloud_cost > 0
        else 0.0
    )

    print("=" * 76)
    print("P3 pilot finished: SV + Cloud, no local execution")
    print("=" * 76)
    print(f"Tasks                         : {n}")
    print(f"Pilot SVs                     : {PILOT_SV}")
    print(f"SV IDs                        : {', '.join(sv_ids)}")
    print(f"Cache gate                    : all services available (pilot only)")
    print(f"omega_T / omega_E             : {OMEGA_T:.2f} / {OMEGA_E:.2f}")
    print(f"V2V max candidate distance    : {V2V_MAX_DISTANCE_M:.1f} m")
    print()
    print(f"Tasks with >=1 eligible SV    : {eligible_tasks}/{n} "
          f"({100.0*eligible_tasks/n:.2f}%)")
    print(f"Slots with >=1 eligible task  : {slots_with_sv_eligible}")
    print(f"Assignments evaluated         : {total_assignments}")
    print(f"Feasible assignments          : {total_feasible_assignments}")
    print(f"Best-solution SLSQP node calls: {best_slsqp_nodes}")
    print()
    print(f"Chosen SV                     : {mode_counts['sv']}/{n} "
          f"({100.0*mode_counts['sv']/n:.2f}%)")
    print(f"Chosen Cloud                  : {mode_counts['cloud']}/{n} "
          f"({100.0*mode_counts['cloud']/n:.2f}%)")
    print(f"Deadline violations           : {deadline_violations}")
    print()
    print(f"Mean task delay               : {total_delay/n:.6f} s")
    print(f"Mean TV transmit energy       : {1000.0*total_energy/n:.6f} mJ")
    print(f"Mean normalized cost J        : {total_cost/n:.6f}")
    print(f"All-cloud mean cost           : {all_cloud_cost/n:.6f}")
    print(f"Cost improvement vs all-cloud : {improvement:.4f}%")
    print()
    print("Chosen node counts:")
    for node, count in sorted(node_counts.items()):
        print(f"  {node}: {count}")
    print()
    print(f"Result file: {out_file}")
    print()
    print("Interpretation:")
    print("- If Cloud is ~100%, the current cloud pilot is too attractive.")
    print("- If SV has a meaningful share, inspect delay/energy before freezing.")
    print("- Cache is intentionally not optimized yet; P1 will be connected next.")


if __name__ == "__main__":
    main()
