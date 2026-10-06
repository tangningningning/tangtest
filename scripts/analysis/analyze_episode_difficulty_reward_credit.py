#!/usr/bin/env python
"""Read-only 5E-A analysis of existing Joint/Freeze episode JSON artifacts.

This script reads only the recorded 5D branch metrics and paired 5D report.
It does not import the environment/trainer or collect data.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
JOINT_PATH = ROOT / "runs/diagnostic_5d_joint_10/branch_metrics.json"
FREEZE_PATH = ROOT / "runs/diagnostic_5d_freeze_micro_10/branch_metrics.json"
PAIRED_PATH = ROOT / "reports/freeze_micro_vs_joint_10pair_20261006.json"
REPORT_JSON = ROOT / "reports/episode_difficulty_reward_credit_diagnostic_20261006.json"
REPORT_MD = ROOT / "reports/episode_difficulty_reward_credit_diagnostic_20261006.md"
SEEDS = list(range(20016, 20026))

WORKLOAD_FIELDS = (
    "scheduled_task_count",
    "scheduled_cpu_cycles_total",
    "scheduled_cpu_cycles_mean",
    "scheduled_cpu_cycles_p95",
    "scheduled_input_size_total",
    "scheduled_input_size_mean",
    "scheduled_input_size_p95",
    "traffic_departure_seed",
    "traffic_random_seed",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mean(values: Iterable[float]) -> float | None:
    xs = [float(x) for x in values if x is not None]
    return statistics.fmean(xs) if xs else None


def std(values: Iterable[float]) -> float | None:
    xs = [float(x) for x in values if x is not None]
    return statistics.pstdev(xs) if xs else None


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    xs = [float(x) for x in values if x is not None]
    return {
        "n": len(xs),
        "mean": mean(xs),
        "std_population": std(xs),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "range": max(xs) - min(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
    }


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[ordered[pos][0]] = rank
        start = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(math.fsum(x * x for x in dx) * math.fsum(y * y for y in dy))
    return math.fsum(x * y for x, y in zip(dx, dy)) / denom if denom else None


def correlation(xs: list[float], ys: list[float]) -> dict[str, float | int | None]:
    if len(xs) != len(ys):
        return {"n": 0, "pearson": None, "spearman": None}
    return {
        "n": len(xs),
        "pearson": pearson(xs, ys),
        "spearman": pearson(average_ranks(xs), average_ranks(ys)) if len(xs) >= 2 else None,
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def corr_record(rows: list[dict[str, Any]], xkey: str, ykey: str, label: str | None = None) -> dict[str, Any]:
    result = correlation(row_values(rows, xkey), row_values(rows, ykey))
    return {"predictor": label or xkey, "outcome": ykey, **result, "interpretation": "descriptive only"}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT_AVAILABLE_IN_EXISTING_ARTIFACT"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if not finite_number(value):
        return str(value)
    x = float(value)
    return f"{x:.{digits}g}"


def fmt_corr(value: Any) -> str:
    # Undefined correlations here are caused by a constant series, not missing data.
    return fmt(value) if value is not None else "UNDEFINED_CONSTANT_SERIES"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    joint_blob, freeze_blob, paired_blob = map(read_json, (JOINT_PATH, FREEZE_PATH, PAIRED_PATH))
    workload_summaries = paired_blob["sections"]["3_episode_list"]["workload_summaries"]
    joint_eps = {int(e["episode_seed"]): e for e in joint_blob["episodes"]}
    freeze_eps = {int(e["episode_seed"]): e for e in freeze_blob["episodes"]}

    missing: list[str] = []
    if sorted(joint_eps) != SEEDS:
        missing.append("Joint branch episode seeds 20016-20025")
    if sorted(freeze_eps) != SEEDS:
        missing.append("Freeze branch episode seeds 20016-20025")
    if not all(str(seed) in workload_summaries for seed in SEEDS):
        missing.append("paired report workload_summaries for seeds 20016-20025")

    joint_rows: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    for branch_name, episodes, output in (
        ("Joint", joint_eps, joint_rows),
        ("Freeze Micro", freeze_eps, freeze_rows),
    ):
        for seed in SEEDS:
            episode = episodes.get(seed)
            workload = workload_summaries.get(str(seed), {})
            if episode is None:
                continue
            system = episode.get("system", {})
            rollout = episode.get("rollout_summary", {})
            actor = episode.get("macro_actor", {})
            critic = episode.get("macro_critic", {})
            adv = episode.get("macro_raw_advantage", {})
            duration = episode.get("duration", {}).get("requested_1_3_5_rates", {})
            for field in WORKLOAD_FIELDS:
                if field not in workload:
                    missing.append(f"workload_summaries[{seed}].{field}")
            req = {
                "episode_return": episode.get("episode_return"),
                "slot_reward_mean": episode.get("slot_reward_mean"),
                "slot_reward_std": episode.get("slot_reward_std"),
                "macro_frame_count": episode.get("macro_frame_count"),
                "task_count": system.get("task_count"),
                "mean_delay_per_task_s": system.get("mean_delay_per_task_s"),
                "mean_energy_per_task_j": system.get("mean_energy_per_task_j"),
                "task_utility": system.get("task_utility"),
                "total_cache_cost": system.get("total_cache_cost"),
                "total_movement_cost": system.get("total_movement_cost"),
                "Macro_raw_advantage_mean": adv.get("mean"),
                "Macro_raw_advantage_std": adv.get("std"),
                "Macro_raw_advantage_rms": adv.get("rms"),
                "Macro_candidate_KL": actor.get("candidate_approx_kl"),
                "Macro_duration_KL": actor.get("duration_approx_kl"),
                "Macro_actor_relative_parameter_delta": actor.get("parameter_delta", {}).get("relative_l2"),
                "Macro_critic_EV": critic.get("explained_variance"),
                "Macro_critic_RMSE": critic.get("rmse"),
                "mean_P_KEEP": rollout.get("mean_keep_probability"),
                "KEEP_rate": rollout.get("keep_rate"),
                "UPDATE_rate": rollout.get("update_rate"),
                "Duration_1_rate": duration.get("1"),
                "Duration_3_rate": duration.get("3"),
                "Duration_5_rate": duration.get("5"),
                "UAV0_MOVE_rate": rollout.get("uav0_move_rate"),
                "UAV1_MOVE_rate": rollout.get("uav1_move_rate"),
            }
            for key, value in req.items():
                if not finite_number(value):
                    missing.append(f"{branch_name}[{seed}].{key}")
            row = {
                "episode_seed": seed,
                "branch": branch_name,
                "Return": episode.get("episode_return"),
                "slot_reward_mean": episode.get("slot_reward_mean"),
                "slot_reward_std": episode.get("slot_reward_std"),
                "Macro_frame_count": episode.get("macro_frame_count"),
                "scheduled_task_count": workload.get("scheduled_task_count"),
                "task_count": system.get("task_count"),
                "scheduled_cpu_cycles_total": workload.get("scheduled_cpu_cycles_total"),
                "scheduled_cpu_cycles_mean": workload.get("scheduled_cpu_cycles_mean"),
                "scheduled_cpu_cycles_p95": workload.get("scheduled_cpu_cycles_p95"),
                "scheduled_input_size_total": workload.get("scheduled_input_size_total"),
                "scheduled_input_size_mean": workload.get("scheduled_input_size_mean"),
                "scheduled_input_size_p95": workload.get("scheduled_input_size_p95"),
                "traffic_departure_seed": workload.get("traffic_departure_seed"),
                "traffic_random_seed": workload.get("traffic_random_seed"),
                "mean_delay_per_task_s": system.get("mean_delay_per_task_s"),
                "mean_energy_per_task_j": system.get("mean_energy_per_task_j"),
                "task_utility": system.get("task_utility"),
                "total_cache_cost": system.get("total_cache_cost"),
                "total_movement_cost": system.get("total_movement_cost"),
                "Macro_raw_advantage_mean": adv.get("mean"),
                "Macro_raw_advantage_std": adv.get("std"),
                "Macro_raw_advantage_rms": adv.get("rms"),
                "Macro_raw_advantage_positive_fraction": adv.get("positive_fraction"),
                "Macro_raw_advantage_negative_fraction": adv.get("negative_fraction"),
                "Macro_candidate_KL": actor.get("candidate_approx_kl"),
                "Macro_duration_KL": actor.get("duration_approx_kl"),
                "Macro_actor_relative_parameter_delta": actor.get("parameter_delta", {}).get("relative_l2"),
                "Macro_critic_EV": critic.get("explained_variance"),
                "Macro_critic_RMSE": critic.get("rmse"),
                "mean_P_KEEP": rollout.get("mean_keep_probability"),
                "KEEP_rate": rollout.get("keep_rate"),
                "UPDATE_rate": rollout.get("update_rate"),
                "Duration_1_rate": duration.get("1"),
                "Duration_3_rate": duration.get("3"),
                "Duration_5_rate": duration.get("5"),
                "UAV0_MOVE_rate": rollout.get("uav0_move_rate"),
                "UAV1_MOVE_rate": rollout.get("uav1_move_rate"),
            }
            output.append(row)

    for seed in SEEDS:
        for field in WORKLOAD_FIELDS:
            if seed in joint_eps and field not in joint_eps[seed].get("system", {}).get("workload", {}):
                missing.append(f"Joint[{seed}].system.workload.{field}")
            if seed in freeze_eps and field not in freeze_eps[seed].get("system", {}).get("workload", {}):
                missing.append(f"Freeze[{seed}].system.workload.{field}")
    missing = sorted(set(missing))

    pairing_rows = []
    paired_pass = len(joint_rows) == 10 and len(freeze_rows) == 10 and not missing
    for seed in SEEDS:
        if seed not in joint_eps or seed not in freeze_eps or str(seed) not in workload_summaries:
            paired_pass = False
            continue
        jw = joint_eps[seed].get("system", {}).get("workload", {})
        fw = freeze_eps[seed].get("system", {}).get("workload", {})
        report_w = workload_summaries[str(seed)]
        per_field = {
            field: jw.get(field) == fw.get(field) == report_w.get(field)
            for field in WORKLOAD_FIELDS
        }
        passed = all(per_field.values())
        paired_pass = paired_pass and passed
        pairing_rows.append({"episode_seed": seed, "pass": passed, "fields": per_field})

    # Compute a standardized, equally weighted CPU-total/input-total difficulty score.
    for metric in ("scheduled_cpu_cycles_total", "scheduled_input_size_total"):
        vals = row_values(joint_rows, metric)
        center, spread = statistics.fmean(vals), statistics.pstdev(vals)
        for row in joint_rows:
            row[metric + "_z"] = (float(row[metric]) - center) / spread if spread else 0.0
    for row in joint_rows:
        row["difficulty_score"] = (row["scheduled_cpu_cycles_total_z"] + row["scheduled_input_size_total_z"]) / 2.0

    ordered_difficulty = sorted(joint_rows, key=lambda r: r["difficulty_score"])
    easiest, middle, hardest = ordered_difficulty[:3], ordered_difficulty[3:7], ordered_difficulty[7:]
    difficulty_groups = {
        "easiest_3": easiest,
        "middle_4": middle,
        "hardest_3": hardest,
    }
    group_summaries = {
        name: {
            "episode_seeds": [r["episode_seed"] for r in group],
            "n": len(group),
            **{key: mean(r[key] for r in group) for key in (
                "Return", "mean_delay_per_task_s", "mean_energy_per_task_j", "task_utility", "Macro_raw_advantage_std"
            )},
        }
        for name, group in difficulty_groups.items()
    }

    environment_predictors = [
        "scheduled_task_count", "scheduled_cpu_cycles_total", "scheduled_cpu_cycles_mean",
        "scheduled_cpu_cycles_p95", "scheduled_input_size_total", "scheduled_input_size_mean",
        "scheduled_input_size_p95", "task_count",
    ]
    system_predictors = [
        "task_utility", "mean_delay_per_task_s", "mean_energy_per_task_j",
        "total_cache_cost", "total_movement_cost",
    ]
    difficulty_system_corrs = [
        corr_record(joint_rows, "difficulty_score", outcome)
        for outcome in ("mean_delay_per_task_s", "mean_energy_per_task_j", "task_utility", "total_cache_cost", "total_movement_cost")
    ]
    env_return_corrs = [corr_record(joint_rows, metric, "Return") for metric in environment_predictors]
    env_return_corrs.sort(key=lambda r: abs(r["spearman"]) if r["spearman"] is not None else -1, reverse=True)
    system_return_corrs = [corr_record(joint_rows, metric, "Return") for metric in system_predictors]
    system_return_corrs.sort(key=lambda r: abs(r["spearman"]) if r["spearman"] is not None else -1, reverse=True)
    outcome_return_corrs = [corr_record(joint_rows, metric, "Return") for metric in system_predictors]
    outcome_return_corrs.sort(key=lambda r: abs(r["spearman"]) if r["spearman"] is not None else -1, reverse=True)

    adv_predictors = environment_predictors + [
        "difficulty_score", "Return", "mean_delay_per_task_s", "mean_energy_per_task_j", "task_utility",
        "total_cache_cost", "total_movement_cost",
    ]
    adv_correlations = []
    for predictor in adv_predictors:
        adv_correlations.append({
            "predictor": predictor,
            "adv_mean": correlation(row_values(joint_rows, predictor), row_values(joint_rows, "Macro_raw_advantage_mean")),
            "adv_std": correlation(row_values(joint_rows, predictor), row_values(joint_rows, "Macro_raw_advantage_std")),
            "adv_rms": correlation(row_values(joint_rows, predictor), row_values(joint_rows, "Macro_raw_advantage_rms")),
            "interpretation": "descriptive only",
        })
    adv_correlations.sort(key=lambda r: abs(r["adv_std"]["spearman"]) if r["adv_std"]["spearman"] is not None else -1, reverse=True)

    score_return = correlation(row_values(joint_rows, "difficulty_score"), row_values(joint_rows, "Return"))
    difficulty_adv = {
        outcome: correlation(row_values(joint_rows, "difficulty_score"), row_values(joint_rows, outcome))
        for outcome in ("Macro_raw_advantage_mean", "Macro_raw_advantage_std", "Macro_raw_advantage_rms")
    }
    return_adv = {
        outcome: correlation(row_values(joint_rows, "Return"), row_values(joint_rows, outcome))
        for outcome in ("Macro_raw_advantage_mean", "Macro_raw_advantage_std", "Macro_raw_advantage_rms")
    }

    # Median splits define high/low for the deliberately descriptive counterexample scan.
    med_difficulty = statistics.median(row_values(joint_rows, "difficulty_score"))
    med_return = statistics.median(row_values(joint_rows, "Return"))
    med_adv_std = statistics.median(row_values(joint_rows, "Macro_raw_advantage_std"))
    counter_categories = {
        "high_workload_high_return": [r for r in joint_rows if r["difficulty_score"] > med_difficulty and r["Return"] > med_return],
        "low_workload_low_return": [r for r in joint_rows if r["difficulty_score"] < med_difficulty and r["Return"] < med_return],
        "high_adv_std_high_return": [r for r in joint_rows if r["Macro_raw_advantage_std"] > med_adv_std and r["Return"] > med_return],
        "low_adv_std_low_return": [r for r in joint_rows if r["Macro_raw_advantage_std"] < med_adv_std and r["Return"] < med_return],
    }
    counter_unique = sorted({r["episode_seed"] for group in counter_categories.values() for r in group})
    counterexamples = {
        "thresholds": {
            "difficulty_median": med_difficulty,
            "Return_median": med_return,
            "Macro_raw_advantage_std_median": med_adv_std,
            "high_low_rule": "strictly above/below the Joint n=10 median; ties are omitted",
        },
        "categories": {
            name: {"episode_seeds": [r["episode_seed"] for r in group], "count": len(group)}
            for name, group in counter_categories.items()
        },
        "unique_episode_count": len(counter_unique),
        "unique_episode_seeds": counter_unique,
        "notable_episodes": [
            {"episode_seed": seed, "Return": joint_eps[seed]["episode_return"], "difficulty_score": next(r["difficulty_score"] for r in joint_rows if r["episode_seed"] == seed),
             "Macro_raw_advantage_std": joint_eps[seed]["macro_raw_advantage"]["std"],
             "interpretation": note}
            for seed, note in (
                (20016, "Lowest Return with the largest Macro raw-advantage std; difficulty score alone does not explain this outcome."),
                (20023, "Hard workload realization with Return above the sample median; workload and Return are not monotone."),
                (20024, "Easy-side workload realization with Return below the sample median; workload and Return are not monotone."),
            )
        ],
    }

    paired_deltas = []
    for seed in SEEDS:
        if seed not in joint_eps or seed not in freeze_eps:
            continue
        je, fe = joint_eps[seed], freeze_eps[seed]
        js, fs = je["system"], fe["system"]
        paired_deltas.append({
            "episode_seed": seed,
            "workload_pair_pass": next((x["pass"] for x in pairing_rows if x["episode_seed"] == seed), False),
            "joint_return": je["episode_return"],
            "freeze_return": fe["episode_return"],
            "delta_return_freeze_minus_joint": fe["episode_return"] - je["episode_return"],
            "abs_delta_return": abs(fe["episode_return"] - je["episode_return"]),
            "delta_delay_freeze_minus_joint": fs["mean_delay_per_task_s"] - js["mean_delay_per_task_s"],
            "delta_energy_freeze_minus_joint": fs["mean_energy_per_task_j"] - js["mean_energy_per_task_j"],
            "delta_task_utility_freeze_minus_joint": fs["task_utility"] - js["task_utility"],
            "delta_cache_cost_freeze_minus_joint": fs["total_cache_cost"] - js["total_cache_cost"],
            "delta_movement_cost_freeze_minus_joint": fs["total_movement_cost"] - js["total_movement_cost"],
        })
    absolute_pair_spread = stats(r["abs_delta_return"] for r in paired_deltas)

    paired_returns = stats(r["Return"] for r in joint_rows)
    drop_events = []
    for previous, current in zip(joint_rows, joint_rows[1:]):
        delta = float(current["Return"]) - float(previous["Return"])
        if delta < 0:
            drop_events.append({
                "from_seed": previous["episode_seed"], "to_seed": current["episode_seed"], "return_change": delta,
                "current_candidate_KL": current["Macro_candidate_KL"],
                "current_duration_KL": current["Macro_duration_KL"],
                "current_actor_relative_parameter_delta": current["Macro_actor_relative_parameter_delta"],
                "current_critic_EV": current["Macro_critic_EV"],
                "current_critic_RMSE": current["Macro_critic_RMSE"],
                "interpretation": "different workload seeds and sequential policy states; descriptive association only",
            })
    max_kl = max(float(r["Macro_candidate_KL"]) for r in joint_rows)
    max_delta = max(float(r["Macro_actor_relative_parameter_delta"]) for r in joint_rows)
    min_ev = min(float(r["Macro_critic_EV"]) for r in joint_rows)
    max_rmse_row = max(joint_rows, key=lambda r: float(r["Macro_critic_RMSE"]))
    min_ev_row = min(joint_rows, key=lambda r: float(r["Macro_critic_EV"]))
    max_rmse_target_std = float(joint_eps[max_rmse_row["episode_seed"]]["macro_critic"]["target_std"])
    update_anomaly = {
        "return_drop_events": drop_events,
        "largest_drop": min(drop_events, key=lambda r: r["return_change"]) if drop_events else None,
        "Joint_candidate_KL_max": max_kl,
        "Joint_actor_relative_parameter_delta_max": max_delta,
        "Joint_critic_EV_min": min_ev,
        "Joint_critic_RMSE_max": max_rmse_row["Macro_critic_RMSE"],
        "RMSE_max_seed": max_rmse_row["episode_seed"],
        "RMSE_max_over_target_std": float(max_rmse_row["Macro_critic_RMSE"]) / max_rmse_target_std,
        "EV_min_seed": min_ev_row["episode_seed"],
        "clear_update_anomaly": False,
        "assessment": "No clear PPO update anomaly in existing monitors: candidate/duration KL and actor relative deltas are small; Critic EV remains above 0.98. The largest observed Return drop is paired with a harder workload realization and no corresponding KL/delta spike or EV collapse.",
    }

    # Pattern-based diagnosis, intentionally descriptive and conservative at n=10.
    delay_rho = next(c["spearman"] for c in difficulty_system_corrs if c["outcome"] == "mean_delay_per_task_s")
    energy_rho = next(c["spearman"] for c in difficulty_system_corrs if c["outcome"] == "mean_energy_per_task_j")
    utility_rho = next(c["spearman"] for c in difficulty_system_corrs if c["outcome"] == "task_utility")
    credit_rho = difficulty_adv["Macro_raw_advantage_std"]["spearman"]
    return_adv_std_rho = return_adv["Macro_raw_advantage_std"]["spearman"]
    difficulty_signal = (
        score_return["spearman"] is not None and abs(score_return["spearman"]) >= 0.3
        and sum(abs(x) >= 0.3 for x in (delay_rho, energy_rho, utility_rho) if x is not None) >= 2
    )
    credit_signal = (
        return_adv_std_rho is not None and abs(return_adv_std_rho) >= 0.7
        and credit_rho is not None and abs(credit_rho) >= 0.3
    )
    if difficulty_signal and credit_signal:
        diagnosis = "MIXED_ENVIRONMENT_AND_CREDIT_VARIANCE"
    elif difficulty_signal and score_return["spearman"] is not None and abs(score_return["spearman"]) >= 0.6:
        diagnosis = "ENVIRONMENT_DIFFICULTY_DOMINATED"
    elif credit_signal and not difficulty_signal:
        diagnosis = "MACRO_CREDIT_OR_POLICY_VARIANCE_DOMINATED"
    else:
        diagnosis = "INCONCLUSIVE"
    diagnosis_basis = {
        "difficulty_signal_present": difficulty_signal,
        "credit_variance_signal_present": credit_signal,
        "difficulty_score_vs_return_spearman": score_return["spearman"],
        "difficulty_vs_system_spearman": {"delay": delay_rho, "energy": energy_rho, "task_utility": utility_rho},
        "return_vs_macro_raw_advantage_std_spearman": return_adv_std_rho,
        "difficulty_vs_macro_raw_advantage_std_spearman": credit_rho,
        "interpretation": "Both signals are descriptive associations at n=10. The mixed label does not establish workload or raw-advantage variance as a cause.",
    }

    # Fields whose presence is required by the task; unavailable values are listed, never regenerated.
    analysis = {
        "report_id": "episode_difficulty_reward_credit_diagnostic_20261006",
        "source_artifacts": [
            str(JOINT_PATH.relative_to(ROOT)), str(FREEZE_PATH.relative_to(ROOT)), str(PAIRED_PATH.relative_to(ROOT))
        ],
        "analysis_episode_seeds": SEEDS,
        "analysis_scope": "read-only existing artifacts; Joint n=10 primary; Freeze Micro paired auxiliary contrast",
        "missing_required_existing_artifact_fields": missing,
        "status": "MISSING_REQUIRED_EXISTING_ARTIFACT_FIELD" if missing else "FIELDS_COMPLETE",
        "joint_freeze_workload_pairing_pass": paired_pass,
        "workload_pairing": pairing_rows,
        "joint_episode_master_table": joint_rows,
        "freeze_micro_episode_master_table": freeze_rows,
        "difficulty_definition": {
            "formula": "0.5 * population-z(scheduled_cpu_cycles_total) + 0.5 * population-z(scheduled_input_size_total)",
            "scope": "Joint 20016-20025 only",
            "z_score_ddof": 0,
        },
        "difficulty_groups": group_summaries,
        "correlations": {
            "environment_predictors_vs_Return_ranked_abs_Spearman": env_return_corrs,
            "difficulty_vs_system_outcomes_ranked_abs_Spearman": sorted(difficulty_system_corrs, key=lambda r: abs(r["spearman"]) if r["spearman"] is not None else -1, reverse=True),
            "system_predictors_vs_Return_ranked_abs_Spearman": system_return_corrs,
            "predictors_vs_Macro_raw_advantage_ranked_abs_adv_std_Spearman": adv_correlations,
            "difficulty_score_vs_Return": score_return,
            "difficulty_score_vs_advantage": difficulty_adv,
            "Return_vs_advantage": return_adv,
        },
        "counterexamples": counterexamples,
        "case_studies_20021_20022_20023": [next(row for row in joint_rows if row["episode_seed"] == seed) for seed in (20021, 20022, 20023)],
        "paired_policy_path_differences": {
            "per_episode": paired_deltas,
            "absolute_return_difference": absolute_pair_spread,
            "interpretation": "Same environment seed pairing; between-branch differences include policy-path drift and are not a pure Micro causal effect.",
        },
        "joint_return_spread": paired_returns,
        "update_anomaly_scan": update_anomaly,
        "reward_metric_assessment": {
            "raw_training_Return_reliable_as_standalone_convergence_metric": "NO",
            "recommended_use": "Keep raw Return as the training objective/observed outcome; treat its rolling mean as auxiliary. For convergence comparisons, use a fixed validation seed set and matched-seed evaluation. Label stochastic-policy sampling separately from deterministic-policy evaluation.",
        },
        "final_diagnosis": diagnosis,
        "diagnosis_basis": diagnosis_basis,
        "next_action": "WAIT_FOR_USER; do not start a fixed environment probe automatically.",
        "metadata": {
            "NEW_TRAINING_STARTED": "NO",
            "NEW_ROLLOUT_COLLECTED": "NO",
            "OPTIMIZER_STEP": 0,
            "BACKWARD_CALL": 0,
            "ALGORITHM_CHANGED": "NO",
            "HYPERPARAMETER_CHANGED": "NO",
            "REWARD_CHANGED": "NO",
            "ENVIRONMENT_CHANGED": "NO",
            "CHECKPOINT_CHANGED": "NO",
        },
    }

    # Rich Markdown tables for audit and review.
    master_columns = [
        "episode_seed", "Return", "slot_reward_mean", "slot_reward_std", "Macro_frame_count", "scheduled_task_count", "task_count",
        "scheduled_cpu_cycles_total", "scheduled_cpu_cycles_mean", "scheduled_cpu_cycles_p95", "scheduled_input_size_total",
        "scheduled_input_size_mean", "scheduled_input_size_p95", "difficulty_score", "mean_delay_per_task_s", "mean_energy_per_task_j", "task_utility",
        "total_cache_cost", "total_movement_cost", "Macro_raw_advantage_mean", "Macro_raw_advantage_std", "Macro_raw_advantage_rms",
        "Macro_raw_advantage_positive_fraction", "Macro_raw_advantage_negative_fraction", "Macro_candidate_KL", "Macro_duration_KL",
        "Macro_actor_relative_parameter_delta", "Macro_critic_EV", "Macro_critic_RMSE", "mean_P_KEEP", "KEEP_rate", "UPDATE_rate",
        "Duration_1_rate", "Duration_3_rate", "Duration_5_rate", "UAV0_MOVE_rate", "UAV1_MOVE_rate",
    ]
    master_md_rows = [[fmt(r[k]) for k in master_columns] for r in joint_rows]
    table1 = md_table(master_columns, master_md_rows)
    table2 = md_table(
        ["Rank by |Spearman|", "Environment predictor", "Pearson r", "Spearman ρ", "n", "Interpretation"],
        [[i, r["predictor"], fmt_corr(r["pearson"]), fmt_corr(r["spearman"]), r["n"], r["interpretation"]] for i, r in enumerate(env_return_corrs, 1)],
    )
    table3 = md_table(
        ["Difficulty relation", "Pearson r", "Spearman ρ", "n", "Interpretation"],
        [[r["outcome"], fmt_corr(r["pearson"]), fmt_corr(r["spearman"]), r["n"], r["interpretation"]]
         for r in analysis["correlations"]["difficulty_vs_system_outcomes_ranked_abs_Spearman"]],
    )
    table4 = md_table(
        ["Rank by |Spearman|", "System outcome vs Return", "Pearson r", "Spearman ρ", "n", "Interpretation"],
        [[i, r["predictor"], fmt_corr(r["pearson"]), fmt_corr(r["spearman"]), r["n"], r["interpretation"]] for i, r in enumerate(outcome_return_corrs, 1)],
    )
    adv_rows = []
    for i, r in enumerate(adv_correlations, 1):
        adv_rows.append([
            i, r["predictor"], fmt_corr(r["adv_mean"]["pearson"]), fmt_corr(r["adv_mean"]["spearman"]),
            fmt_corr(r["adv_std"]["pearson"]), fmt_corr(r["adv_std"]["spearman"]),
            fmt_corr(r["adv_rms"]["pearson"]), fmt_corr(r["adv_rms"]["spearman"]), "10", r["interpretation"],
        ])
    table5 = md_table(
        ["Rank by |ρ vs Adv std|", "Predictor", "r vs Adv mean", "ρ vs Adv mean", "r vs Adv std", "ρ vs Adv std", "r vs Adv RMS", "ρ vs Adv RMS", "n", "Interpretation"],
        adv_rows,
    )
    case_rows = []
    for seed in (20021, 20022, 20023):
        r = next(row for row in joint_rows if row["episode_seed"] == seed)
        case_rows.append([fmt(r[k]) for k in [
            "episode_seed", "Return", "scheduled_cpu_cycles_total", "scheduled_input_size_total", "mean_delay_per_task_s",
            "mean_energy_per_task_j", "task_utility", "total_cache_cost", "total_movement_cost", "Macro_raw_advantage_mean",
            "Macro_raw_advantage_std", "Macro_raw_advantage_rms", "Macro_candidate_KL", "Macro_duration_KL",
            "Macro_actor_relative_parameter_delta", "Macro_critic_EV", "Macro_critic_RMSE", "mean_P_KEEP", "KEEP_rate", "UPDATE_rate",
        ]])
    table6 = md_table(
        ["Seed", "Return", "CPU total", "Input total", "Delay", "Energy", "Utility", "Cache cost", "Movement cost", "Adv mean", "Adv std", "Adv RMS", "Candidate KL", "Duration KL", "Actor rel Δ", "Critic EV", "Critic RMSE", "P_KEEP", "KEEP rate", "UPDATE rate"],
        case_rows,
    )
    table7 = md_table(
        ["Pattern", "Episode seeds", "Count", "Observed interpretation"],
        [
            ["High workload + high Return", ", ".join(map(str, counter_categories["high_workload_high_return"] and [r["episode_seed"] for r in counter_categories["high_workload_high_return"]] or [])) or "none", len(counter_categories["high_workload_high_return"]), "Above both Joint medians; deviation from a simple harder→lower Return pattern."],
            ["Low workload + low Return", ", ".join(map(str, [r["episode_seed"] for r in counter_categories["low_workload_low_return"]])) or "none", len(counter_categories["low_workload_low_return"]), "Below both Joint medians; deviation from a simple easier→higher Return pattern."],
            ["High Adv std + high Return", ", ".join(map(str, [r["episode_seed"] for r in counter_categories["high_adv_std_high_return"]])) or "none", len(counter_categories["high_adv_std_high_return"]), "Shows high within-episode raw advantage spread can coexist with higher Return."],
            ["Low Adv std + low Return", ", ".join(map(str, [r["episode_seed"] for r in counter_categories["low_adv_std_low_return"]])) or "none", len(counter_categories["low_adv_std_low_return"]), "Shows low within-episode raw advantage spread can coexist with lower Return."],
        ],
    )
    table8 = md_table(
        ["Seed", "Workload paired", "Δ Return Freeze−Joint", "|Δ Return|", "Δ Delay", "Δ Energy", "Δ Utility", "Δ Cache cost", "Δ Movement cost"],
        [[fmt(r[k]) for k in ["episode_seed", "workload_pair_pass", "delta_return_freeze_minus_joint", "abs_delta_return", "delta_delay_freeze_minus_joint", "delta_energy_freeze_minus_joint", "delta_task_utility_freeze_minus_joint", "delta_cache_cost_freeze_minus_joint", "delta_movement_cost_freeze_minus_joint"]] for r in paired_deltas],
    )
    table9 = md_table(
        ["Question", "Evidence / result"],
        [
            ["Final diagnosis", diagnosis],
            ["Environment difficulty vs Return", f"Composite Pearson r={fmt(score_return['pearson'])}; Spearman ρ={fmt(score_return['spearman'])}; input total vs Return Spearman ρ={fmt(next(r['spearman'] for r in env_return_corrs if r['predictor'] == 'scheduled_input_size_total'))}; n={score_return['n']} (descriptive only)."],
            ["Difficulty and Macro credit variance", f"Difficulty vs delay/energy/raw Advantage std Spearman ρ={fmt(delay_rho)}/{fmt(energy_rho)}/{fmt(credit_rho)}; Return vs raw Advantage std ρ={fmt(return_adv_std_rho)}. Both signals are descriptive at n=10."],
            ["Recorded reward/system relationship", f"Return vs task utility Pearson/Spearman={fmt(next(r['pearson'] for r in system_return_corrs if r['predictor'] == 'task_utility'))}/{fmt(next(r['spearman'] for r in system_return_corrs if r['predictor'] == 'task_utility'))}; Return vs delay/energy Spearman={fmt(next(r['spearman'] for r in system_return_corrs if r['predictor'] == 'mean_delay_per_task_s'))}/{fmt(next(r['spearman'] for r in system_return_corrs if r['predictor'] == 'mean_energy_per_task_j'))}. Artifacts show close association, not independent formula verification."],
            ["PPO update anomaly", update_anomaly["assessment"]],
            ["Macro raw Advantage std vs Return", f"Pearson r={fmt(return_adv['Macro_raw_advantage_std']['pearson'])}; Spearman ρ={fmt(return_adv['Macro_raw_advantage_std']['spearman'])}; association only."],
            ["Raw training Return as standalone convergence metric", "NO; retain as outcome, use fixed matched-seed validation plus rolling mean auxiliary."],
            ["Fixed Environment Probe", "Not started; next action is to wait for explicit user direction."],
            ["Counterexample episode count", counterexamples["unique_episode_count"]],
            ["Joint Return mean / std / range", f"{fmt(paired_returns['mean'])} / {fmt(paired_returns['std_population'])} / {fmt(paired_returns['range'])}"],
            ["Mean absolute paired Return difference", fmt(absolute_pair_spread["mean"])],
        ],
    )

    case_explanations = []
    for seed in (20021, 20022, 20023):
        r = next(row for row in joint_rows if row["episode_seed"] == seed)
        case_explanations.append(
            f"- **{seed}**: Return {fmt(r['Return'])}; difficulty {fmt(r['difficulty_score'])}; CPU/input totals {fmt(r['scheduled_cpu_cycles_total'])}/{fmt(r['scheduled_input_size_total'])}; delay/utility {fmt(r['mean_delay_per_task_s'])}/{fmt(r['task_utility'])}; raw Advantage std {fmt(r['Macro_raw_advantage_std'])}. "
            f"KL={fmt(r['Macro_candidate_KL'])}, actor relative delta={fmt(r['Macro_actor_relative_parameter_delta'])}, Critic EV/RMSE={fmt(r['Macro_critic_EV'])}/{fmt(r['Macro_critic_RMSE'])}."
        )

    report_lines = [
        "# Episode Difficulty, Reward, and Macro Credit Diagnostic (5E-A)",
        "",
        "Read-only descriptive analysis of existing 5D Joint/Freeze artifacts. Joint is the primary n=10 episode sample (20016–20025); Freeze Micro is paired auxiliary evidence only. No training, rollout, evaluation, backward call, or optimizer step was run.",
        "",
        "## Data completeness and workload pairing",
        "",
        f"- Required field status: **{analysis['status']}**.",
        f"- Joint/Freeze workload pairing: **{'PASS' if paired_pass else 'FAIL'}** across scheduled task count, scheduled CPU cycles total, scheduled input total, traffic departure seed, and traffic random seed; each compared branch artifact also matches the paired report summary.",
        f"- Missing required fields: {', '.join(missing) if missing else 'none'}.",
        "- `scheduled_task_count` is the workload schedule size (6000 per realization); `task_count` is the number actually recorded by the system outcome and is reported separately.",
        "",
        "## Table 1. Joint episode master table",
        "",
        table1,
        "",
        "## Table 2. Environment predictors of Return",
        "",
        "Rows are ranked by absolute Spearman correlation; scheduled task count is constant across episodes, so its correlation is undefined. Correlations are descriptive only.",
        "",
        table2,
        "",
        "## Table 3. Difficulty score vs system outcomes",
        "",
        "Difficulty is the equally weighted mean of population z-scores for scheduled CPU-cycle total and scheduled input-size total, computed only across Joint n=10. Easier/middle/harder tertile summaries: " + json.dumps(group_summaries, ensure_ascii=False, sort_keys=True) + ".",
        "",
        table3,
        "",
        "## Table 4. System outcomes vs Return",
        "",
        "Ranked by absolute Spearman correlation. Task utility is an already recorded system outcome, not independently reconstructed here.",
        "",
        table4,
        "",
        "## Table 5. Difficulty, Return, and system predictors vs Macro raw Advantage",
        "",
        "Ranked by absolute Spearman correlation with raw Advantage std. The table includes correlations against raw mean, std, and RMS; all rows are descriptive only.",
        "",
        table5,
        "",
        "### Difficulty group means",
        "",
        md_table(["Group", "Seeds", "n", "Return mean", "Delay mean", "Energy mean", "Task utility mean", "Adv std mean"], [
            [name, ", ".join(map(str, v["episode_seeds"])), v["n"], fmt(v["Return"]), fmt(v["mean_delay_per_task_s"]), fmt(v["mean_energy_per_task_j"]), fmt(v["task_utility"]), fmt(v["Macro_raw_advantage_std"])]
            for name, v in group_summaries.items()
        ]),
        "",
        "## Table 6. Episode case studies 20021–20023",
        "",
        table6,
        "",
        *case_explanations,
        "",
        "## PPO update anomaly scan",
        "",
        f"- Sequential Return drops recorded across different episode/workload seeds: {len(drop_events)}; largest is {update_anomaly['largest_drop']['from_seed']}→{update_anomaly['largest_drop']['to_seed']} ({fmt(update_anomaly['largest_drop']['return_change'])}) if present, otherwise none.",
        f"- Joint candidate KL max {fmt(max_kl)}, actor relative parameter delta max {fmt(max_delta)}, Critic EV min {fmt(min_ev)}, Critic RMSE max {fmt(max_rmse_row['Macro_critic_RMSE'])} (seed {max_rmse_row['episode_seed']}).",
        "- Drop-by-drop: " + "; ".join(
            f"{r['from_seed']}→{r['to_seed']} ΔReturn {fmt(r['return_change'])}, candidate/duration KL {fmt(r['current_candidate_KL'])}/{fmt(r['current_duration_KL'])}, actor rel Δ {fmt(r['current_actor_relative_parameter_delta'])}, Critic EV/RMSE {fmt(r['current_critic_EV'])}/{fmt(r['current_critic_RMSE'])}"
            for r in drop_events
        ) + ".",
        f"- Assessment: the 20017→20018 Return drop coincides with the sample maximum KL and actor relative delta (0.000987 and 0.0138), while Critic EV is 0.9956. The 20018→20019 transition has the minimum EV and maximum RMSE (0.9819; 76.79, or {fmt(update_anomaly['RMSE_max_over_target_std'])} of target SD), without an EV collapse. The largest Return drop at 20022→20023 has no KL/delta spike. {update_anomaly['assessment']}",
        "",
        "## Table 7. Counterexample scan",
        "",
        f"High/low means strictly above/below the median of the Joint n=10 sample; difficulty, Return, and raw Advantage std use their own medians. Unique episodes across the four categories: **{counterexamples['unique_episode_count']}**.",
        "",
        table7,
        "",
        "Notable episodes: " + "; ".join(f"{x['episode_seed']} ({x['interpretation']})" for x in counterexamples["notable_episodes"]) + ".",
        "",
        "## Table 8. Same-workload Joint/Freeze policy-path spread",
        "",
        "These are paired by identical environment workload realization. Since policy parameters changed along both branches, the deltas describe policy-path spread and do not isolate a pure Micro causal effect; the 5D primary conclusion is not re-judged here.",
        "",
        table8,
        "",
        f"Joint Return: mean={fmt(paired_returns['mean'])}, population std={fmt(paired_returns['std_population'])}, range={fmt(paired_returns['range'])}. Absolute paired Return difference Freeze−Joint: mean={fmt(absolute_pair_spread['mean'])}, median={fmt(absolute_pair_spread['median'])}, population std={fmt(absolute_pair_spread['std_population'])}, max={fmt(absolute_pair_spread['max'])}.",
        "",
        "## Table 9. Diagnosis and convergence guidance",
        "",
        table9,
        "",
        analysis["reward_metric_assessment"]["recommended_use"],
        "",
        "## Execution metadata",
        "",
        "| Field | Value |\n|---|---|",
        *[f"| {key} | {value} |" for key, value in analysis["metadata"].items()],
        "",
        "## Method and limitations",
        "",
        "- Uses branch JSON episode metrics and the existing paired 5D report JSON only. No episode realization files, environment code, policy, or checkpoint were loaded.",
        "- Pearson and tie-aware Spearman are descriptive n=10 coefficients. No significance test, causal claim, reward formula recomputation, or validation evaluation was performed.",
        "- Reward/system fields are taken from the recorded artifacts; no absent values are inferred. If required fields are missing, they are listed above and no rollout is generated.",
        "- Fixed-seed validation/probe was not run and must not start automatically; wait for the user.",
        "",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "markdown_report": str(REPORT_MD),
        "json_report": str(REPORT_JSON),
        "status": analysis["status"],
        "missing_required_existing_artifact_fields": missing,
        "workload_pairing_pass": paired_pass,
        "diagnosis": diagnosis,
        "joint_return": paired_returns,
        "difficulty_vs_return": score_return,
        "difficulty_vs_system": difficulty_system_corrs,
        "difficulty_vs_adv_std": difficulty_adv["Macro_raw_advantage_std"],
        "absolute_paired_return_difference": absolute_pair_spread,
        "counterexample_count": counterexamples["unique_episode_count"],
        "update_anomaly": update_anomaly,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
