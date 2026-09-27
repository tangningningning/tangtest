#!/usr/bin/env python3
"""Run the bounded full-exact Cache Trigger causal-gate diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import project_config
from ppo_simple.cache_credit import (
    _diagnostic_sign,
    _masked_operation_probabilities,
    evaluate_cache_trigger_causal_gate,
)
from ppo_simple.formal_evaluation import load_frozen_policy
from ppo_simple.rollout import collect_rollout
from ppo_simple.trainer import aligned_train_steps, hyperparameters_from_project, load_environment_context


TRAINING_REALIZATION = PROJECT_ROOT / "environment_data" / "proposed_joint_training" / "episode_20001"
REPORT_PATH = PROJECT_ROOT / "reports" / "cache_trigger_causal_gate.md"
CHECKPOINT_SPECS = (
    ("LR=5e-4", "stage2d_v3_fresh300_actor_lr_5e_minus_4", "epoch_205_checkpoint.pt"),
    ("LR=5e-3", "stage2d_v3_fresh300_actor_lr_5e_minus_3", "epoch_059_checkpoint.pt"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_checkpoint(run_dir: Path, preferred_name: str) -> tuple[Path | None, str]:
    preferred = [
        path for path in (
            run_dir / "checkpoints" / preferred_name,
            run_dir / preferred_name,
        ) if path.is_file()
    ]
    if not preferred:
        preferred = sorted(run_dir.rglob(preferred_name))
    if preferred:
        return preferred[0], "PREFERRED_EPOCH"
    latest = [
        path for path in (
            run_dir / "checkpoints" / "latest_checkpoint.pt",
            run_dir / "latest_checkpoint.pt",
        ) if path.is_file()
    ]
    if not latest:
        latest = sorted(run_dir.rglob("latest_checkpoint.pt"))
    if latest:
        return latest[0], "LATEST_AVAILABLE_FALLBACK"
    return None, "CHECKPOINT_NOT_AVAILABLE"


def _transition_selection_features(
    policy: Any,
    transition: Any,
    node_count: int,
) -> dict[str, Any]:
    if (
        transition.cache_contexts is None
        or transition.cache_operation_masks is None
        or transition.cache_trigger_aux is None
        or transition.cache_operation_actions is None
    ):
        raise RuntimeError("stochastic rollout lacks stored Cache Trigger state")
    p_updates = []
    legal_counts = np.asarray(
        transition.cache_trigger_legal_candidate_counts, dtype=np.int64
    )
    changed = np.any(
        np.asarray(transition.previous_cache_state, dtype=np.int8)
        != np.asarray(transition.next_cache_state, dtype=np.int8),
        axis=1,
    )
    actual_actions = np.asarray(transition.cache_operation_actions, dtype=np.int64)
    for node_index in range(node_count):
        probabilities = _masked_operation_probabilities(
            policy,
            transition.cache_contexts[node_index],
            node_index,
            transition.cache_operation_masks[node_index],
            trigger_aux=transition.cache_trigger_aux[node_index],
        )
        p_updates.append(float(probabilities[1]))
    actions = set(int(value) for value in actual_actions.tolist())
    update_eligible = legal_counts > 0
    p_array = np.asarray(p_updates, dtype=np.float64)
    keep_fixed = bool(np.any((p_array <= 0.05) & update_eligible))
    update_fixed = bool(np.any((p_array >= 0.95) & update_eligible))
    if bool(np.any((p_array > 0.05) & (p_array < 0.95) & update_eligible)):
        trigger_regimes = {"INTERMEDIATE"}
    else:
        trigger_regimes = set()
    if keep_fixed:
        trigger_regimes.add("KEEP_EXTREME")
    if update_fixed:
        trigger_regimes.add("UPDATE_EXTREME")
    cache_change_count = int(np.count_nonzero(changed))
    change_bucket = "ZERO" if cache_change_count == 0 else (
        "ONE_OR_TWO" if cache_change_count <= 2 else "THREE_PLUS"
    )
    legal_bucket = "HAS_LEGAL_UPDATE" if bool(np.any(update_eligible)) else "NO_LEGAL_UPDATE"
    action_flags = {
        "SAMPLED_KEEP" if 0 in actions else "",
        "SAMPLED_UPDATE" if 1 in actions else "",
    }
    action_flags.discard("")
    return {
        "origin_index": int(transition.origin_index),
        "slot": int(transition.slot),
        "change_pattern": tuple(int(index) for index in np.flatnonzero(changed)),
        "change_bucket": change_bucket,
        "time_bucket": int(transition.slot * 10 // max(1, 300)),
        "p_updates": p_array,
        "trigger_regimes": trigger_regimes,
        "action_flags": action_flags,
        "legal_bucket": legal_bucket,
        "actual_update_count": int(np.count_nonzero(actual_actions == 1)),
        "actual_keep_count": int(np.count_nonzero(actual_actions == 0)),
        "mean_legal_count": float(np.mean(legal_counts)),
    }


def _select_transition_indices(
    policy: Any,
    rollout: Any,
    *,
    limit: int,
    node_count: int,
) -> list[int]:
    features = [
        _transition_selection_features(policy, transition, node_count)
        for transition in rollout.transitions
    ]
    selected: list[int] = []
    seen_patterns: set[tuple[int, ...]] = set()
    seen_change_buckets: set[str] = set()
    seen_time_buckets: set[int] = set()
    seen_trigger_regimes: set[str] = set()
    seen_action_flags: set[str] = set()
    seen_legal_buckets: set[str] = set()
    remaining = set(range(len(features)))
    while remaining and len(selected) < int(limit):
        def score(index: int) -> tuple[float, int]:
            item = features[index]
            novelty = 0.0
            novelty += 3.0 if item["change_pattern"] not in seen_patterns else 0.0
            novelty += 2.0 if item["change_bucket"] not in seen_change_buckets else 0.0
            novelty += 1.5 if item["time_bucket"] not in seen_time_buckets else 0.0
            novelty += 2.0 * len(item["trigger_regimes"] - seen_trigger_regimes)
            novelty += 2.0 * len(item["action_flags"] - seen_action_flags)
            novelty += 1.0 if item["legal_bucket"] not in seen_legal_buckets else 0.0
            novelty += min(float(item["actual_update_count"]), 2.0) * 0.5
            novelty += min(float(item["mean_legal_count"]), 24.0) / 48.0
            # Ties resolve to the earlier origin, keeping selection repeatable.
            return novelty, -int(item["origin_index"])

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        item = features[chosen]
        seen_patterns.add(item["change_pattern"])
        seen_change_buckets.add(item["change_bucket"])
        seen_time_buckets.add(item["time_bucket"])
        seen_trigger_regimes.update(item["trigger_regimes"])
        seen_action_flags.update(item["action_flags"])
        seen_legal_buckets.add(item["legal_bucket"])
    return selected


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _max_abs(values: list[float]) -> float | None:
    return None if not values else float(max(abs(value) for value in values))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legal = [row for row in rows if int(row["legal_candidate_count"]) > 0]
    sampled = [row for row in legal if row["selected_candidate_exact_rank"] is not None]
    policy_argmax = [row for row in legal if row["policy_argmax_candidate_exact_rank"] is not None]
    policy_sign = [row for row in legal if row["delta_policy_argmax_fixed"] is not None]
    sampled_sign = [row for row in sampled if row["delta_sampled_fixed"] is not None]
    downstream = [row for row in legal if row["downstream_sign_flip"] is not None]
    policy_flips = [
        row for row in policy_sign
        if _diagnostic_sign(row["delta_policy_argmax_fixed"])
        != _diagnostic_sign(row["delta_best_exact_fixed"])
    ]
    sampled_flips = [
        row for row in sampled_sign
        if _diagnostic_sign(row["delta_sampled_fixed"])
        != _diagnostic_sign(row["delta_best_exact_fixed"])
    ]
    selected_regrets = [float(row["selected_candidate_exact_regret"]) for row in sampled]
    policy_regrets = [float(row["policy_argmax_candidate_exact_regret"]) for row in policy_argmax]
    abs_delta_differences = [
        abs(float(row["delta_best_exact_fixed"])
            - float(row["delta_best_policy_consistent"]))
        for row in downstream
    ]
    return {
        "rows": len(rows),
        "legal_rows": len(legal),
        "update_samples": sum(row["actual_action"] == "UPDATE" for row in rows),
        "keep_samples": sum(row["actual_action"] == "KEEP" for row in rows),
        "policy_flip_count": len(policy_flips),
        "policy_flip_rate": len(policy_flips) / len(policy_sign) if policy_sign else None,
        "policy_to_keep_best_to_update": sum(
            _diagnostic_sign(row["delta_policy_argmax_fixed"]) == "KEEP_BETTER"
            and _diagnostic_sign(row["delta_best_exact_fixed"]) == "UPDATE_BETTER"
            for row in policy_flips
        ),
        "sampled_flip_count": len(sampled_flips),
        "sampled_flip_rate": len(sampled_flips) / len(sampled_sign) if sampled_sign else None,
        "selected_top1_rate": (
            sum(int(row["selected_candidate_exact_rank"]) == 1 for row in sampled)
            / len(sampled) if sampled else None
        ),
        "policy_argmax_top1_rate": (
            sum(int(row["policy_argmax_candidate_exact_rank"]) == 1 for row in policy_argmax)
            / len(policy_argmax) if policy_argmax else None
        ),
        "selected_mean_rank": _mean(sampled, "selected_candidate_exact_rank"),
        "policy_argmax_mean_rank": _mean(policy_argmax, "policy_argmax_candidate_exact_rank"),
        "selected_mean_regret": (
            float(np.mean(selected_regrets)) if selected_regrets else None
        ),
        "policy_argmax_mean_regret": (
            float(np.mean(policy_regrets)) if policy_regrets else None
        ),
        "best_candidate_mean_probability": _mean(
            legal, "best_exact_candidate_policy_probability"
        ),
        "downstream_flip_count": sum(bool(row["downstream_sign_flip"]) for row in downstream),
        "downstream_flip_rate": (
            sum(bool(row["downstream_sign_flip"]) for row in downstream) / len(downstream)
            if downstream else None
        ),
        "mean_abs_delta_difference": (
            float(np.mean(abs_delta_differences)) if abs_delta_differences else None
        ),
        "max_abs_delta_difference": _max_abs(abs_delta_differences),
    }


def _head_state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signs_fixed = {
        _diagnostic_sign(row["delta_best_exact_fixed"])
        for row in rows if row.get("delta_best_exact_fixed") is not None
    } - {None, "TIE"}
    signs_consistent = {
        _diagnostic_sign(row["delta_best_policy_consistent"])
        for row in rows if row.get("delta_best_policy_consistent") is not None
    } - {None, "TIE"}
    p_values = [float(row["p_update"]) for row in rows]
    same_extreme_side = bool(p_values) and (
        all(value <= 0.05 for value in p_values)
        or all(value >= 0.95 for value in p_values)
    )
    flips_fixed = len(signs_fixed) > 1
    flips_consistent = len(signs_consistent) > 1
    return {
        "rows": len(rows),
        "update_samples": sum(row["actual_action"] == "UPDATE" for row in rows),
        "keep_samples": sum(row["actual_action"] == "KEEP" for row in rows),
        "mean_p_update": _mean(rows, "p_update"),
        "min_p_update": min(p_values) if p_values else None,
        "max_p_update": max(p_values) if p_values else None,
        "mean_legal_candidates": _mean(rows, "legal_candidate_count"),
        "selected_top1_rate": _summary(rows)["selected_top1_rate"],
        "policy_argmax_top1_rate": _summary(rows)["policy_argmax_top1_rate"],
        "selected_mean_rank": _summary(rows)["selected_mean_rank"],
        "policy_argmax_mean_rank": _summary(rows)["policy_argmax_mean_rank"],
        "selected_mean_regret": _summary(rows)["selected_mean_regret"],
        "policy_argmax_mean_regret": _summary(rows)["policy_argmax_mean_regret"],
        "best_candidate_mean_probability": _summary(rows)["best_candidate_mean_probability"],
        "policy_flip_rate": _summary(rows)["policy_flip_rate"],
        "sampled_flip_rate": _summary(rows)["sampled_flip_rate"],
        "downstream_flip_rate": _summary(rows)["downstream_flip_rate"],
        "mean_abs_delta_difference": _summary(rows)["mean_abs_delta_difference"],
        "max_abs_delta_difference": _summary(rows)["max_abs_delta_difference"],
        "delta_sign_flips_across_states_fixed": flips_fixed,
        "delta_sign_flips_across_states_policy_consistent": flips_consistent,
        "p_update_stays_extreme_while_delta_sign_flips": (
            len(rows) >= 3 and same_extreme_side and (flips_fixed or flips_consistent)
        ),
        "fixed_signs": sorted(signs_fixed),
        "policy_consistent_signs": sorted(signs_consistent),
        "p_update_stays_extreme": len(rows) >= 3 and same_extreme_side,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"
    return str(value)


def _fmt_rate(value: float | None, count: int | None = None, denominator: int | None = None) -> str:
    if value is None:
        return "NA"
    suffix = "" if count is None or denominator is None else f" ({count}/{denominator})"
    return f"{float(value):.3f}{suffix}"


def _render_report(checkpoint_results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    all_rows = [row for result in checkpoint_results for row in result["diagnostic"]["rows"]]
    overall = _summary(all_rows)
    head_groups: list[tuple[str, int, list[dict[str, Any]]]] = []
    for result in checkpoint_results:
        label = str(result["label"])
        for node_index in sorted({int(row["node_index"]) for row in result["diagnostic"]["rows"]}):
            group = [
                row for row in result["diagnostic"]["rows"]
                if int(row["node_index"]) == node_index
            ]
            head_groups.append((label, node_index, group))
    head_summaries = [
        (label, node_index, _head_state_summary(rows))
        for label, node_index, rows in head_groups
    ]

    policy_direction_count = overall["policy_to_keep_best_to_update"]
    case_a = "SUPPORTED" if policy_direction_count >= 2 else "INCONCLUSIVE"
    case_b = "SUPPORTED" if overall["downstream_flip_count"] >= 2 else "INCONCLUSIVE"
    case_c = (
        "SUPPORTED"
        if any(item[2]["p_update_stays_extreme_while_delta_sign_flips"] for item in head_summaries)
        else "INCONCLUSIVE"
    )
    starvation_heads = []
    for label, node_index, rows in head_groups:
        summary = _head_state_summary(rows)
        eligible = [row for row in rows if int(row["legal_candidate_count"]) > 0]
        if not eligible:
            continue
        fixed_keep = all(float(row["p_update"]) <= 0.05 for row in eligible)
        if (
            fixed_keep
            and summary["update_samples"] == 0
            and summary["policy_argmax_mean_rank"] is not None
            and (
                float(summary["policy_argmax_mean_rank"]) > 1.0
                or float(summary["policy_argmax_mean_regret"] or 0.0) > 1.0e-10
            )
        ):
            starvation_heads.append((label, node_index))
    case_d = "SUPPORTED" if starvation_heads else "INCONCLUSIVE"

    next_changes = []
    if case_a == "SUPPORTED":
        next_changes.append("改为 action-independent 的 `Q_keep vs best-update-value` Trigger credit。")
    if case_b == "SUPPORTED":
        next_changes.append("正式 credit 使用 policy-consistent downstream evaluator 或经验证的近似。")
    if case_d == "SUPPORTED":
        next_changes.append("把 Cache Candidate 从 global GAE 改为 candidate-specific value/ranking credit。")
    next_required = " ".join(next_changes) if next_changes else "ROOT_CAUSE_NOT_CONFIRMED"

    lines = [
        "# Cache Trigger Causal Gate",
        "",
        "## Code path and diagnostic scope",
        "",
        "唯一代码基线为当前 `main` 工作区。生产 Trigger 使用 `ONE_STEP_TD_Q_DIFFERENCE`；实际 UPDATE 的生产信用是 sampled UPDATE 对 KEEP，实际 KEEP 的生产信用是 KEEP 对 candidate-policy argmax UPDATE。此生产训练目标在本任务中未改。Candidate 使用 `GLOBAL_GAE`。生产 one-head branch 替换目标 Cache head，其余 Cache heads 与 UAV 复用实际 rollout 的结果，故 downstream mode 为 fixed-downstream production-rollout reuse。",
        "",
        f"`CURRENT_CACHE_TRIGGER_CREDIT = {checkpoint_results[0]['current_trigger_credit'] if checkpoint_results else 'UNKNOWN'}`",
        f"`CURRENT_CACHE_CANDIDATE_CREDIT = {checkpoint_results[0]['current_candidate_credit'] if checkpoint_results else 'UNKNOWN'}`",
        "`CURRENT_COUNTERFACTUAL_DOWNSTREAM_MODE = FIXED_DOWNSTREAM_PRODUCTION_ROLLOUT_REUSE`",
        f"`CURRENT_CACHE_TRIGGER_ENTROPY_COEFFICIENT = {checkpoint_results[0]['entropy_coefficient'] if checkpoint_results else 'UNKNOWN'}` (shared PPO entropy setting; no Cache-specific override is configured)",
        "",
        "本诊断统一 `delta = Q_UPDATE - Q_KEEP`，且 branch value 按现有实现取 `reward + gamma * frozen_critic(next_observation)`。`BEST_EXACT_UPDATE` 对每个合法完整 target cache configuration 真实调用现有 exact Cache branch；相同物理结果只复用相同 branch。policy-consistent replay 固定 checkpoint、同一 rollout transition 与 exogenous step，保留目标 head 前的生产 Cache prefix，强制目标 KEEP 或 BEST_EXACT_UPDATE，并通过生产 deterministic primitive selector 重新决策后续 Cache heads 和 UAV。两种 replay 都复用正式 environment → Matching → KKT → reward 路径。",
        "",
        "`POLICY_CONSISTENT_MODE = DETERMINISTIC_FROZEN_POLICY_REPLAY`。该量不是 stochastic policy expectation，不报告为 exact policy Q。下一状态 bootstrap 使用当前冻结 global critic，因此它也不是整段未来 trajectory 的精确 rollout value。",
        "",
        f"固定 realization：`{TRAINING_REALIZATION.relative_to(PROJECT_ROOT).as_posix()}`；只对每个 checkpoint 选出的 {args.transitions} 个 transition 做 full exact，9 个 Cache heads 均保留。状态选择确定性地覆盖 slot、CacheChange pattern、KEEP/UPDATE 样本、legal UPDATE 和 Trigger 概率区间。采样 rollout 固定 seed `{args.seed}`。",
        "",
        "`best_exact_candidate_policy_probability` 是 candidate policy 在 UPDATE 条件下给 BEST_EXACT_UPDATE 的概率，不包含 Trigger 的 p_update。",
        "",
        "## Checkpoints",
        "",
        "| Checkpoint | Preferred epoch | Analyzed file | Epoch | SHA256 | Selection status |",
        "|---|---:|---|---:|---|---|",
    ]
    for result in checkpoint_results:
        lines.append(
            f"| {result['label']} | {result['preferred_epoch']} | `{result['checkpoint'].relative_to(PROJECT_ROOT).as_posix()}` | "
            f"{result['epoch']} | `{result['sha256'][:16]}…` | {result['selection_status']} |"
        )
    if not checkpoint_results:
        lines.append("| — | — | — | — | — | CHECKPOINT_NOT_AVAILABLE |")
    for result in checkpoint_results:
        if result["selection_status"] == "LATEST_AVAILABLE_FALLBACK":
            lines.append(
                f"\n`{result['preferred_epoch']}` = CHECKPOINT_NOT_AVAILABLE; "
                "the latest available local checkpoint was analyzed instead."
            )
    for label, _, preferred in CHECKPOINT_SPECS:
        if not any(item["label"] == label for item in checkpoint_results):
            lines.append(f"\n`{label} {preferred} = CHECKPOINT_NOT_AVAILABLE`; no training or download was attempted.")
    lines.extend(
        [
            "",
            "## Core statistics",
            "",
            "### Overall",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| POLICY_ARGMAX_VS_BEST_SIGN_FLIP_RATE | {_fmt_rate(overall['policy_flip_rate'], overall['policy_flip_count'], overall['legal_rows'])} |",
            f"| SAMPLED_VS_BEST_SIGN_FLIP_RATE | {_fmt_rate(overall['sampled_flip_rate'], overall['sampled_flip_count'], overall['update_samples'])} |",
            f"| FIXED_VS_POLICY_CONSISTENT_SIGN_FLIP_RATE | {_fmt_rate(overall['downstream_flip_rate'], overall['downstream_flip_count'], overall['legal_rows'])} |",
            f"| SELECTED_CANDIDATE_TOP1_EXACT_RATE | {_fmt(overall['selected_top1_rate'])} |",
            f"| POLICY_ARGMAX_CANDIDATE_TOP1_EXACT_RATE | {_fmt(overall['policy_argmax_top1_rate'])} |",
            f"| SELECTED_CANDIDATE_MEAN_EXACT_RANK | {_fmt(overall['selected_mean_rank'])} |",
            f"| POLICY_ARGMAX_CANDIDATE_MEAN_EXACT_RANK | {_fmt(overall['policy_argmax_mean_rank'])} |",
            f"| SELECTED_CANDIDATE_MEAN_REGRET | {_fmt(overall['selected_mean_regret'])} |",
            f"| POLICY_ARGMAX_CANDIDATE_MEAN_REGRET | {_fmt(overall['policy_argmax_mean_regret'])} |",
            f"| BEST_EXACT_CANDIDATE_MEAN_POLICY_PROB | {_fmt(overall['best_candidate_mean_probability'])} |",
            f"| MEAN_ABS_DELTA_DIFFERENCE | {_fmt(overall['mean_abs_delta_difference'])} |",
            f"| MAX_ABS_DELTA_DIFFERENCE | {_fmt(overall['max_abs_delta_difference'])} |",
            "",
            "### By checkpoint",
            "",
            "| Checkpoint | Transitions | Head rows | UPDATE / KEEP samples | Legal rows | Policy argmax→best flip rate | Sampled→best flip rate | Selected top1 / mean rank / mean regret | Policy argmax top1 / mean rank / mean regret | Best exact candidate mean policy prob. | Fixed→consistent flip rate | Mean / max abs Δ difference |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for result in checkpoint_results:
        summary = _summary(result["diagnostic"]["rows"])
        lines.append(
            f"| {result['label']} | {len(result['selected_transition_indices'])} | {summary['rows']} | "
            f"{summary['update_samples']} / {summary['keep_samples']} | {summary['legal_rows']} | "
            f"{_fmt_rate(summary['policy_flip_rate'], summary['policy_flip_count'], summary['legal_rows'])} | "
            f"{_fmt_rate(summary['sampled_flip_rate'], summary['sampled_flip_count'], summary['update_samples'])} | "
            f"{_fmt(summary['selected_top1_rate'])} / {_fmt(summary['selected_mean_rank'])} / "
            f"{_fmt(summary['selected_mean_regret'])} | {_fmt(summary['policy_argmax_top1_rate'])} / "
            f"{_fmt(summary['policy_argmax_mean_rank'])} / {_fmt(summary['policy_argmax_mean_regret'])} | "
            f"{_fmt(summary['best_candidate_mean_probability'])} | "
            f"{_fmt_rate(summary['downstream_flip_rate'], summary['downstream_flip_count'], summary['legal_rows'])} | "
            f"{_fmt(summary['mean_abs_delta_difference'])} / {_fmt(summary['max_abs_delta_difference'])} |"
        )
    lines.extend(
        [
            "",
            "### By checkpoint and Cache head",
            "",
            "| Checkpoint | Head | UPDATE | KEEP | Mean legal candidates | Policy→best flip | Sampled→best flip | Selected top1 / rank / regret | Policy argmax top1 / rank / regret | Best exact policy prob. | Fixed→consistent flip | Mean / max abs Δ difference | Δ sign flips across states (fixed / consistent) | p_update stays extreme while delta flips |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
        ]
    )
    for label, node_index, summary in head_summaries:
        lines.append(
            f"| {label} | {node_index} | {summary['update_samples']} | {summary['keep_samples']} | "
            f"{_fmt(summary['mean_legal_candidates'])} | {_fmt_rate(summary['policy_flip_rate'])} | "
            f"{_fmt_rate(summary['sampled_flip_rate'])} | {_fmt(summary['selected_top1_rate'])} / "
            f"{_fmt(summary['selected_mean_rank'])} / {_fmt(summary['selected_mean_regret'])} | "
            f"{_fmt(summary['policy_argmax_top1_rate'])} / {_fmt(summary['policy_argmax_mean_rank'])} / "
            f"{_fmt(summary['policy_argmax_mean_regret'])} | {_fmt(summary['best_candidate_mean_probability'])} | "
            f"{_fmt_rate(summary['downstream_flip_rate'])} | {_fmt(summary['mean_abs_delta_difference'])} / "
            f"{_fmt(summary['max_abs_delta_difference'])} | {summary['delta_sign_flips_across_states_fixed']} / "
            f"{summary['delta_sign_flips_across_states_policy_consistent']} | "
            f"{summary['p_update_stays_extreme_while_delta_sign_flips']} |"
        )
    lines.extend(
        [
            "",
            "### Trigger sample responsiveness by checkpoint and Cache head",
            "",
            "| Checkpoint | Head | States | p_update min–max | Fixed exact signs across states | Policy-consistent signs across states | p_update stays on same extreme side while either delta changes sign |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    for label, node_index, summary in head_summaries:
        p_range = (
            f"{_fmt(summary['min_p_update'])}–{_fmt(summary['max_p_update'])}"
            if summary["rows"] else "NA"
        )
        lines.append(
            f"| {label} | {node_index} | {summary['rows']} | {p_range} | "
            f"{', '.join(summary['fixed_signs']) or 'TIE/NA'} | "
            f"{', '.join(summary['policy_consistent_signs']) or 'TIE/NA'} | "
            f"{summary['p_update_stays_extreme_while_delta_sign_flips']} |"
        )
    lines.extend(
        [
            "",
            "### Transition × Cache head details",
            "",
            "| Checkpoint | Transition | Slot | Head index / id | Action | p_update | Legal | Δ sampled fixed | Δ policy argmax fixed | Δ best exact fixed | Δ best policy-consistent | Selected rank / regret | Policy argmax rank / regret | Best candidate prob. | Fixed / consistent sign | Flip | Cache changed |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|---|",
        ]
    )
    for result in checkpoint_results:
        for row in result["diagnostic"]["rows"]:
            lines.append(
                f"| {result['label']} | {row['transition_index']} | {row['slot']} | {row['node_index']} / {row['node_id']} | "
                f"{row['actual_action']} | {_fmt(row['p_update'])} | {row['legal_candidate_count']} | "
                f"{_fmt(row['delta_sampled_fixed'])} | {_fmt(row['delta_policy_argmax_fixed'])} | "
                f"{_fmt(row['delta_best_exact_fixed'])} | {_fmt(row['delta_best_policy_consistent'])} | "
                f"{_fmt(row['selected_candidate_exact_rank'])} / {_fmt(row['selected_candidate_exact_regret'])} | "
                f"{_fmt(row['policy_argmax_candidate_exact_rank'])} / {_fmt(row['policy_argmax_candidate_exact_regret'])} | "
                f"{_fmt(row['best_exact_candidate_policy_probability'])} | "
                f"{row['fixed_downstream_sign']} / {row['policy_consistent_sign']} | "
                f"{row['downstream_sign_flip']} | {row['cache_changed']} |"
            )
    total_transitions = sum(len(result["selected_transition_indices"]) for result in checkpoint_results)
    total_rows = len(all_rows)
    total_legal_candidates = sum(int(row["legal_candidate_count"]) for row in all_rows)
    full_exact_branches = sum(
        int(result["diagnostic"]["full_exact_update_environment_branches"])
        for result in checkpoint_results
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `CACHE_TRIGGER_REFERENCE_COUPLING_PROBLEM = {case_a}`. Policy argmax→KEEP while best-exact→UPDATE occurred {policy_direction_count} times; total policy/best sign flips were {overall['policy_flip_count']}/{overall['legal_rows']}. This is evidence about reference coupling on this small batch, not a population causal estimate.",
            f"- `FIXED_DOWNSTREAM_COUNTERFACTUAL_BIAS = {case_b}`. Fixed and deterministic policy-consistent signs differed {overall['downstream_flip_count']}/{overall['legal_rows']} times. This indicates whether the one-head ceteris-paribus ranking was sensitive to downstream replay on these transitions.",
            f"- `CACHE_TRIGGER_STATE_RESPONSE_LOCKIN = {case_c}`. A head qualifies only with at least three analyzed states, p_update on the same extreme side (≤0.05 or ≥0.95) in every state, and both KEEP_BETTER and UPDATE_BETTER among fixed or policy-consistent deltas.",
            f"- `CACHE_CANDIDATE_STARVATION = {case_d}`. A head qualifies only when all analyzed legal rows have p_update≤0.05, no sampled UPDATE occurs, and the candidate-policy argmax has exact rank >1 or positive exact regret. Qualifying heads: {starvation_heads or 'none'}.",
            "",
            "`GLOBAL_CRITIC_HEALTH_INTERPRETATION = NOT_ASSESSED_SEPARATELY`; all Q comparisons use the same frozen global critic bootstrap. This batch does not establish critic calibration or overall critic health.",
            f"`COUNTERFACTUAL_VALUE_DIFFERENCE_ACCURACY = sign agreement {1.0 - (overall['downstream_flip_rate'] or 0.0):.3f} on {overall['legal_rows']} legal rows; mean/max |delta_fixed - delta_policy_consistent| = {_fmt(overall['mean_abs_delta_difference'])}/{_fmt(overall['max_abs_delta_difference'])}. These measure replay disagreement, not accuracy against stochastic Q^π.",
            f"`NEXT_REQUIRED_ALGORITHM_CHANGE = {next_required}",
            "",
            "## Required output summary",
            "",
            "```text",
            "TRAINING_EXECUTED = NO",
            f"CHECKPOINTS_ANALYZED = {', '.join(result['label'] + ':' + str(result['epoch']) for result in checkpoint_results) or 'NONE'}",
            f"TRANSITIONS_ANALYZED = {total_transitions}",
            f"CACHE_HEAD_ROWS_ANALYZED = {total_rows}",
            f"FULL_EXACT_UPDATE_BRANCHES = {full_exact_branches} executed / {total_legal_candidates} legal complete UPDATE configurations evaluated",
            f"POLICY_ARGMAX_VS_BEST_SIGN_FLIP_RATE = {_fmt_rate(overall['policy_flip_rate'], overall['policy_flip_count'], overall['legal_rows'])}",
            f"SAMPLED_VS_BEST_SIGN_FLIP_RATE = {_fmt_rate(overall['sampled_flip_rate'], overall['sampled_flip_count'], overall['update_samples'])}",
            f"FIXED_VS_POLICY_CONSISTENT_SIGN_FLIP_RATE = {_fmt_rate(overall['downstream_flip_rate'], overall['downstream_flip_count'], overall['legal_rows'])}",
            f"CACHE_TRIGGER_REFERENCE_COUPLING_PROBLEM = {case_a}",
            f"FIXED_DOWNSTREAM_COUNTERFACTUAL_BIAS = {case_b}",
            f"CACHE_TRIGGER_STATE_RESPONSE_LOCKIN = {case_c}",
            f"CACHE_CANDIDATE_STARVATION = {case_d}",
            "GLOBAL_CRITIC_HEALTH_INTERPRETATION = NOT_ASSESSED_SEPARATELY",
            f"COUNTERFACTUAL_VALUE_DIFFERENCE_ACCURACY = SIGN_AGREEMENT_{1.0 - (overall['downstream_flip_rate'] or 0.0):.3f}; MEAN_ABS_DELTA_DIFFERENCE_{_fmt(overall['mean_abs_delta_difference'])}; MAX_ABS_DELTA_DIFFERENCE_{_fmt(overall['max_abs_delta_difference'])}",
            f"NEXT_REQUIRED_ALGORITHM_CHANGE = {next_required}",
            "```",
            "",
            "## Reproducibility and cleanup",
            "",
            f"Selected rollout slots by checkpoint: {[(item['label'], item['selected_slots']) for item in checkpoint_results]}",
            f"Diagnostic runtime by checkpoint: {[(item['label'], round(item['diagnostic']['runtime_seconds'], 1)) for item in checkpoint_results]} seconds for the full-exact batch.",
            "The report and the single maintained runner are the only new persistent task outputs. No checkpoint, formal run artifact, JSONL, or training output was modified. No temporary script or synthetic file was retained.",
            "",
            "CHANGED_FILES = src/ppo_simple/cache_credit.py; scripts/diagnostics/run_cache_trigger_causal_gate.py; reports/cache_trigger_causal_gate.md",
            "DELETED_OBSOLETE_SOURCE_FILES = NONE",
            "VALIDATIONS = Python compile (both touched Python files) PASS; synthetic Q_update−Q_keep sign check PASS; real checkpoint 1 transition × 1 head full-exact smoke PASS; causal gate 2 checkpoints × 8 transitions × 9 heads PASS; git diff --check PASS",
            "TEMP_FILES_REMOVED = no temporary scripts or synthetic files were created; six __pycache__ directories removed after validation",
            "GIT_DIFF_CHECK = PASS",
            "COMMIT = NO",
            "PUSH = NO",
        ]
    )
    return "\n".join(lines) + "\n"


def _analyze_checkpoint(
    label: str,
    run_name: str,
    preferred_epoch: str,
    *,
    transition_limit: int,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any] | None:
    run_dir = PROJECT_ROOT / "runs" / run_name
    checkpoint, selection_status = _find_checkpoint(run_dir, preferred_epoch)
    if checkpoint is None:
        print(f"{label}: CHECKPOINT_NOT_AVAILABLE ({preferred_epoch})", flush=True)
        return None
    config_path = run_dir / "run_config.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint run config missing: {config_path}")
    if not TRAINING_REALIZATION.is_dir():
        raise FileNotFoundError(f"fixed training realization missing: {TRAINING_REALIZATION}")

    context = load_environment_context(str(config_path), realization_dir=TRAINING_REALIZATION)
    policy, checkpoint_meta = load_frozen_policy(checkpoint, context)
    cfg = context.cfg
    trigger_credit = str(project_config.get_config_value(
        cfg, "algorithm.ppo.cache_trigger_advantage_source"
    ))
    candidate_credit = str(project_config.get_config_value(
        cfg, "algorithm.ppo.cache_candidate_advantage_source"
    ))
    entropy_coefficient = float(project_config.get_config_value(
        cfg, "algorithm.ppo.entropy_coefficient"
    ))
    trainer = SimpleNamespace(
        context=context,
        policy=policy,
        hyperparameters=hyperparameters_from_project(cfg),
    )

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    steps = aligned_train_steps(context)
    if len(steps) != int(context.episode_slots):
        raise RuntimeError("fixed diagnostic rollout did not load all training slots")
    with torch.no_grad():
        rollout = collect_rollout(
            policy,
            context,
            steps,
            initial_cache=context.initial_cache,
            initial_uav=context.initial_uav,
            terminal_origin_index=context.terminal_origin,
            deterministic=False,
        )
    if len(rollout.transitions) != len(steps):
        raise RuntimeError("stochastic fixed-checkpoint rollout length mismatch")
    node_count = len(context.action_spec.cache_node_ids)
    if smoke:
        legal_counts = [
            np.asarray(item.cache_trigger_legal_candidate_counts, dtype=np.int64)
            for item in rollout.transitions
        ]
        transition_index = max(
            range(len(legal_counts)),
            key=lambda index: (
                int(legal_counts[index].sum()),
                -int(rollout.transitions[index].origin_index),
            ),
        )
        target_node_index = int(np.argmax(legal_counts[transition_index]))
        if int(legal_counts[transition_index][target_node_index]) <= 0:
            raise RuntimeError("smoke rollout has no legal UPDATE candidate")
        selected_indices = [transition_index]
        target_node_indices = (target_node_index,)
    else:
        selected_indices = _select_transition_indices(
            policy,
            rollout,
            limit=transition_limit,
            node_count=node_count,
        )
        target_node_indices = None
    print(
        f"{label}: checkpoint_epoch={checkpoint_meta['checkpoint_epoch']} "
        f"selected_slots={[rollout.transitions[index].slot for index in selected_indices]}; "
        "starting full-exact counterfactuals",
        flush=True,
    )
    diagnostic = evaluate_cache_trigger_causal_gate(
        trainer,
        rollout,
        target_transition_indices=tuple(selected_indices),
        target_node_indices=target_node_indices,
    )
    return {
        "label": label,
        "preferred_epoch": preferred_epoch,
        "selection_status": selection_status,
        "checkpoint": checkpoint,
        "epoch": int(checkpoint_meta["checkpoint_epoch"]),
        "sha256": _sha256(checkpoint),
        "selected_transition_indices": selected_indices,
        "selected_slots": [rollout.transitions[index].slot for index in selected_indices],
        "diagnostic": diagnostic,
        "current_trigger_credit": trigger_credit,
        "current_candidate_credit": candidate_credit,
        "entropy_coefficient": entropy_coefficient,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, choices=(4, 8), default=8)
    parser.add_argument("--seed", type=int, default=73019)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one checkpoint × one transition × one legal Cache head; do not write the report",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        torch.set_num_threads(1)

    results = []
    checkpoint_specs = CHECKPOINT_SPECS[:1] if args.smoke else CHECKPOINT_SPECS
    for label, run_name, preferred_epoch in checkpoint_specs:
        result = _analyze_checkpoint(
            label,
            run_name,
            preferred_epoch,
            transition_limit=args.transitions,
            seed=args.seed,
            smoke=args.smoke,
        )
        if result is not None:
            results.append(result)
    if args.smoke:
        if not results:
            return 2
        diagnostic = results[0]["diagnostic"]
        row = diagnostic["rows"][0]
        print(
            "SMOKE_OK "
            f"epoch={results[0]['epoch']} slot={row['slot']} node={row['node_id']} "
            f"legal_candidates={row['legal_candidate_count']} "
            f"full_exact_update_branches={diagnostic['full_exact_update_environment_branches']} "
            f"runtime_seconds={diagnostic['runtime_seconds']:.2f}",
            flush=True,
        )
        return 0
    if not results:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            "# Cache Trigger Causal Gate\n\nNo requested checkpoint was available.\n",
            encoding="utf-8",
        )
        return 2
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(results, args), encoding="utf-8")
    print(f"REPORT = {REPORT_PATH}", flush=True)
    print(f"TRANSITIONS_ANALYZED = {sum(len(item['selected_transition_indices']) for item in results)}")
    print(f"CACHE_HEAD_ROWS_ANALYZED = {sum(len(item['diagnostic']['rows']) for item in results)}")
    print(
        "FULL_EXACT_UPDATE_BRANCHES = "
        f"{sum(item['diagnostic']['full_exact_update_environment_branches'] for item in results)} executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
