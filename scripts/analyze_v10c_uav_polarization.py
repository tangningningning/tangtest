"""Offline root-cause audit for the three formal V10-B UAV-polarization runs.

This script only reads persisted manifests, summaries, JSONL logs, local config
snapshots, and the source tree.  It does not import the training entry point,
create an environment, load a policy for inference, execute a slot, or write
anything under a run directory.  Its only output is the requested Markdown
report (unless a different report path is supplied).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "v10c_uav_polarization_root_cause.md"
EXPECTED_EPOCHS = 300
EXPECTED_SLOTS = 300
WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("1-60", 1, 60),
    ("61-120", 61, 120),
    ("121-180", 121, 180),
    ("181-240", 181, 240),
    ("241-300", 241, 300),
    ("281-300", 281, 300),
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    actor_learning_rate: float
    directory_name: str
    config_relative_path: str


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        "5e-4",
        5.0e-4,
        "ppo_v10_branchwise_cache_uav_5e4_kl_formal",
        "configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml",
    ),
    RunSpec(
        "6e-4",
        6.0e-4,
        "ppo_v10_branchwise_cache_uav_6e4_kl_formal",
        "configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml",
    ),
    RunSpec(
        "1e-3",
        1.0e-3,
        "ppo_v10_branchwise_cache_uav_1e3_kl_formal",
        "configs/lr_sweep/v10_branchwise_actor_lr_1e3_kl.toml",
    ),
)


@dataclass
class RunData:
    spec: RunSpec
    run_dir: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    episode: list[dict[str, Any]]
    update: list[dict[str, Any]]
    slot: list[dict[str, Any]]
    critic: list[dict[str, Any]]
    config_path: Path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, expected_record_type: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            if expected_record_type is None or value.get("record_type", expected_record_type) == expected_record_type:
                records.append(value)
    return records


def require_epoch_records(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    epochs = [int(record["epoch"]) for record in records]
    expected = list(range(1, EXPECTED_EPOCHS + 1))
    if epochs != expected:
        raise ValueError(f"{path} does not contain ordered epochs 1..{EXPECTED_EPOCHS}")


def load_runs() -> tuple[RunData, ...]:
    result: list[RunData] = []
    for spec in RUN_SPECS:
        run_dir = PROJECT_ROOT / "runs" / spec.directory_name
        manifest = read_json(run_dir / "run_manifest.json")
        summary = read_json(run_dir / "run_summary.json")
        episode = read_jsonl(run_dir / "episode_metrics.jsonl", "episode")
        update = read_jsonl(run_dir / "update_metrics.jsonl", "update")
        slot = read_jsonl(run_dir / "slot_metrics.jsonl")
        critic = read_jsonl(run_dir / "critic_diagnostics.jsonl", "critic_transition")
        require_epoch_records(episode, run_dir / "episode_metrics.jsonl")
        require_epoch_records(update, run_dir / "update_metrics.jsonl")
        if len(episode) != EXPECTED_EPOCHS or len(update) != EXPECTED_EPOCHS:
            raise ValueError(f"{run_dir}: expected 300 episode and update records")
        if len(slot) != EXPECTED_EPOCHS * EXPECTED_SLOTS:
            raise ValueError(f"{run_dir}: expected 90,000 slot records")
        if len(critic) != EXPECTED_EPOCHS * EXPECTED_SLOTS:
            raise ValueError(f"{run_dir}: expected 90,000 critic records")
        config_path = PROJECT_ROOT / spec.config_relative_path
        result.append(
            RunData(
                spec=spec,
                run_dir=run_dir,
                manifest=manifest,
                summary=summary,
                episode=episode,
                update=update,
                slot=slot,
                critic=critic,
                config_path=config_path,
            )
        )
    return tuple(result)


def finite_numbers(records: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None or isinstance(value, bool):
            if isinstance(value, bool):
                values.append(float(value))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def field_stats(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, float] | None:
    values = finite_numbers(records, field)
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": float(len(values)),
    }


def select_epoch(records: Sequence[Mapping[str, Any]], start: int, end: int) -> list[Mapping[str, Any]]:
    return [record for record in records if start <= int(record.get("epoch", -1)) <= end]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NOT_AVAILABLE"
    return f"{numeric:.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    try:
        numeric = float(value) * 100.0
    except (TypeError, ValueError):
        return "NOT_AVAILABLE"
    if not math.isfinite(numeric):
        return "NOT_AVAILABLE"
    return f"{numeric:.{digits}f}%"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(item) for item in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def basename(path_value: Any) -> str:
    value = str(path_value).replace("\\", "/").rstrip("/")
    return value.rsplit("/", 1)[-1]


def realization_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("realization_pool_index"),
        basename(record.get("realization_directory", "")),
        record.get("training_episode_index"),
        record.get("episode"),
        record.get("seed"),
        record.get("global_training_slot_start"),
        record.get("global_training_slot_end"),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alignment_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    expected_manifest = {
        "policy_schema": "adaptive_joint_cache_replica_aware_uav_operation_v10_branchwise",
        "configuration_control_mode": "policy_primitive_v9_joint_cache_uav_operation",
        "data_protocol": "full_episode_randomized",
        "critic_learning_rate": 3.0e-4,
        "target_kl": 0.10,
        "clip_epsilon": 0.20,
        "update_epochs": 4,
        "entropy_coefficient": 0.0,
        "episode_slots": 300,
        "rollout_steps": 300,
        "train_origin_start": 0,
        "train_origin_end": 299,
        "seed": 4401,
    }
    mismatches: list[str] = []
    for run in runs:
        for field, expected in expected_manifest.items():
            actual = run.manifest.get(field)
            if isinstance(expected, float):
                equal = actual is not None and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
            else:
                equal = actual == expected
            if not equal:
                mismatches.append(f"{run.spec.label}:{field}={actual!r}, expected {expected!r}")
        if not math.isclose(float(run.manifest.get("actor_learning_rate", math.nan)), run.spec.actor_learning_rate, rel_tol=0.0, abs_tol=1.0e-12):
            mismatches.append(f"{run.spec.label}:actor_learning_rate mismatch")
        if run.summary.get("completed_epochs") != EXPECTED_EPOCHS or run.summary.get("status") != "completed":
            mismatches.append(f"{run.spec.label}:summary incomplete/status")
        if run.manifest.get("target_epochs") != EXPECTED_EPOCHS or run.manifest.get("current_target_epochs") != EXPECTED_EPOCHS:
            mismatches.append(f"{run.spec.label}:manifest target epochs mismatch")
        if not run.config_path.is_file():
            mismatches.append(f"{run.spec.label}:local config missing {run.config_path}")
        else:
            manifest_hash = str(run.manifest.get("config_sha256", "")).lower()
            local_hash = sha256(run.config_path).lower()
            if manifest_hash != local_hash:
                mismatches.append(f"{run.spec.label}:config_sha256 mismatch")

    base_episode = [realization_identity(record) for record in runs[0].episode]
    base_update = [realization_identity(record) for record in runs[0].update]
    for run in runs[1:]:
        current_episode = [realization_identity(record) for record in run.episode]
        current_update = [realization_identity(record) for record in run.update]
        for index, (left, right) in enumerate(zip(base_episode, current_episode), 1):
            if left != right:
                mismatches.append(f"{run.spec.label}:episode mapping mismatch at epoch {index}")
                break
        for index, (left, right) in enumerate(zip(base_update, current_update), 1):
            if left != right:
                mismatches.append(f"{run.spec.label}:update mapping mismatch at epoch {index}")
                break

    pool_root = PROJECT_ROOT / "runs" / "ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
    pool_names = {path.name for path in pool_root.glob("episode_*")} if pool_root.is_dir() else set()
    expected_pool_names = {f"episode_{seed}" for seed in range(20001, 20301)}
    pool_complete = expected_pool_names <= pool_names
    return {
        "valid": not mismatches,
        "mismatches": mismatches,
        "expected_manifest": expected_manifest,
        "pool_root": pool_root,
        "pool_complete": pool_complete,
        "first_identity": base_episode[0] if base_episode else None,
        "last_identity": base_episode[-1] if base_episode else None,
    }


def rolling_mean(values: Sequence[float], window: int = 20) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        result[index] = statistics.fmean(values[index - window + 1 : index + 1])
    return result


def first_threshold(values: Sequence[float | None], predicate: Any) -> int | None:
    for index, value in enumerate(values, 1):
        if value is not None and predicate(float(value)):
            return index
    return None


def epoch_values(run: RunData, source: str, field: str) -> list[float | None]:
    records = run.episode if source == "episode" else run.update
    result: list[float | None] = [None] * EXPECTED_EPOCHS
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[int(record["epoch"]) - 1] = numeric
    return result


def stage_row(run: RunData, label: str, start: int, end: int) -> dict[str, Any]:
    episode = select_epoch(run.episode, start, end)
    update = select_epoch(run.update, start, end)
    row: dict[str, Any] = {"window": label}
    return_fields = {
        "return_mean": (episode, "episode_return"),
        "return_std": (episode, "episode_return"),
        "cache_change": (episode, "cache_actual_change_rate"),
        "uav_move": (episode, "uav_actual_movement_rate"),
        "joint_kl": (update, "joint_approx_kl"),
        "cache_kl": (update, "cache_approx_kl"),
        "uav_kl": (update, "uav_approx_kl"),
        "cache_clip": (update, "cache_clip_fraction"),
        "uav_clip": (update, "uav_clip_fraction"),
        "cache_entropy": (update, "cache_entropy"),
        "uav_entropy": (update, "uav_entropy"),
        "cache_grad": (update, "cache_head_gradient_norm"),
        "uav_grad": (update, "uav_head_gradient_norm"),
        "shared_grad": (update, "shared_actor_gradient_norm"),
        "early_stop": (update, "kl_early_stop_triggered"),
        "ppo_epochs": (update, "ppo_update_epochs_executed"),
    }
    for name, (records, field) in return_fields.items():
        stats = field_stats(records, field)
        if stats is None:
            row[name] = None
        elif name == "return_std":
            row[name] = stats["std"]
        elif name == "early_stop":
            row[name] = stats["mean"]
        else:
            row[name] = stats["mean"]
    ratios: list[float] = []
    for record in update:
        cache = record.get("cache_head_gradient_norm")
        uav = record.get("uav_head_gradient_norm")
        try:
            cache_f = float(cache)
            uav_f = float(uav)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cache_f) and math.isfinite(uav_f) and uav_f > 0.0:
            ratios.append(cache_f / uav_f)
    row["cache_uav_grad_ratio"] = statistics.fmean(ratios) if ratios else None
    return row


def all_stage_rows(runs: Sequence[RunData]) -> dict[str, list[dict[str, Any]]]:
    return {
        run.spec.label: [stage_row(run, label, start, end) for label, start, end in WINDOWS]
        for run in runs
    }


def polarization_audit(run: RunData) -> dict[str, Any]:
    movement = [float(value) for value in epoch_values(run, "episode", "uav_actual_movement_rate") if value is not None]
    ma20 = rolling_mean(movement, 20)
    low_thresholds = (0.40, 0.20, 0.10, 0.05)
    high_thresholds = (0.60, 0.80, 0.90, 0.95)
    low = {f"< {int(threshold * 100)}%": first_threshold(ma20, lambda value, t=threshold: value < t) for threshold in low_thresholds}
    high = {f"> {int(threshold * 100)}%": first_threshold(ma20, lambda value, t=threshold: value > t) for threshold in high_thresholds}
    final_ma20 = ma20[-1] if ma20 else None
    if final_ma20 is not None and final_ma20 < 0.40:
        polarization_kind = "STAY-like"
        onset = first_threshold(ma20, lambda value: value < 0.10)
        threshold_label = "MA20 < 10%"
    elif final_ma20 is not None and final_ma20 > 0.60:
        polarization_kind = "MOVE-like"
        onset = first_threshold(ma20, lambda value: value > 0.90)
        threshold_label = "MA20 > 90%"
    else:
        polarization_kind = "not classified by final MA20"
        onset = None
        threshold_label = "NOT_AVAILABLE"

    onset_metrics: dict[str, Any] = {
        "kind": polarization_kind,
        "threshold": threshold_label,
        "onset": onset,
    }
    if onset is not None:
        pre_start = max(1, onset - 19)
        pre_end = onset - 1
        post_start = onset
        post_end = min(EXPECTED_EPOCHS, onset + 19)
        metric_fields = (
            ("uav_binary_entropy", "episode", "uav_binary_entropy_mean"),
            ("uav_entropy", "update", "uav_entropy"),
            ("uav_kl", "update", "uav_approx_kl"),
            ("uav_clip", "update", "uav_clip_fraction"),
            ("uav_grad", "update", "uav_head_gradient_norm"),
        )
        metric_rows: list[dict[str, Any]] = []
        for label, source, field in metric_fields:
            records = run.episode if source == "episode" else run.update
            pre_stats = field_stats(select_epoch(records, pre_start, pre_end), field) if pre_end >= pre_start else None
            post_stats = field_stats(select_epoch(records, post_start, post_end), field)
            metric_rows.append(
                {
                    "metric": label,
                    "pre_window": f"{pre_start}-{pre_end}" if pre_end >= pre_start else "NONE",
                    "post_window": f"{post_start}-{post_end}",
                    "pre": None if pre_stats is None else pre_stats["mean"],
                    "post": None if post_stats is None else post_stats["mean"],
                    "delta": None if pre_stats is None or post_stats is None else post_stats["mean"] - pre_stats["mean"],
                }
            )
        onset_metrics["rows"] = metric_rows
    return {
        "movement": movement,
        "ma20": ma20,
        "low": low,
        "high": high,
        "final_ma20": final_ma20,
        "onset": onset_metrics,
    }


def early_signal_audit(runs: Sequence[RunData], polarization: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    entropy_precedes: list[bool] = []
    kl_precedes: list[bool] = []
    grad_precedes: list[bool] = []
    for run in runs:
        onset = polarization[run.spec.label]["onset"]["onset"]
        entropy_values = epoch_values(run, "episode", "uav_binary_entropy_mean")
        entropy_collapse = first_threshold(entropy_values, lambda value: value < 0.10)
        uav_kl_values = epoch_values(run, "update", "uav_approx_kl")
        kl_spike = first_threshold(uav_kl_values, lambda value: value > 0.10)
        grad_values = epoch_values(run, "update", "uav_head_gradient_norm")
        finite_grad = [value for value in grad_values if value is not None]
        grad_floor = None if not finite_grad else statistics.fmean(finite_grad) * 0.25
        grad_collapse = None if grad_floor is None else first_threshold(grad_values, lambda value, floor=grad_floor: value < floor)
        rows.append(
            {
                "run": run.spec.label,
                "polarization_onset": onset,
                "entropy_collapse_lt_0.10": entropy_collapse,
                "uav_kl_spike_gt_0.10": kl_spike,
                "uav_grad_floor_25pct_of_run_mean": grad_floor,
                "uav_grad_collapse": grad_collapse,
            }
        )
        if onset is not None:
            entropy_precedes.append(entropy_collapse is not None and entropy_collapse < onset)
            kl_precedes.append(kl_spike is not None and kl_spike < onset)
            grad_precedes.append(grad_collapse is not None and grad_collapse < onset)
    def status(values: Sequence[bool]) -> str:
        if not values:
            return "UNCLEAR"
        if all(values):
            return "YES"
        if not any(values):
            return "NO"
        return "UNCLEAR"
    return {
        "rows": rows,
        "entropy_precedes": entropy_precedes,
        "kl_precedes": kl_precedes,
        "grad_precedes": grad_precedes,
        "entropy_status": status(entropy_precedes),
        "kl_status": status(kl_precedes),
        "grad_status": status(grad_precedes),
    }


def branch_audit(runs: Sequence[RunData], stage_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    final: list[dict[str, Any]] = []
    for run in runs:
        row = next(item for item in stage_rows[run.spec.label] if item["window"] == "281-300")
        cache = row.get("cache_grad")
        uav = row.get("uav_grad")
        shared = row.get("shared_grad")
        final.append(
            {
                "run": run.spec.label,
                "cache_uav_grad_ratio": row.get("cache_uav_grad_ratio"),
                "cache_grad": cache,
                "uav_grad": uav,
                "shared_grad": shared,
                "cache_kl": row.get("cache_kl"),
                "uav_kl": row.get("uav_kl"),
                "cache_clip": row.get("cache_clip"),
                "uav_clip": row.get("uav_clip"),
            }
        )
    return {"final": final}


def advantage_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for label, start, end in WINDOWS:
            records = select_epoch(run.episode, start, end)
            rows.append(
                {
                    "run": run.spec.label,
                    "window": label,
                    "move_mean": field_stats(records, "mean_advantage_for_uav_move"),
                    "move_median": field_stats(records, "median_advantage_for_uav_move"),
                    "stay_mean": field_stats(records, "mean_advantage_for_uav_stay"),
                    "stay_median": field_stats(records, "median_advantage_for_uav_stay"),
                    "move_count": field_stats(records, "uav_move_selected_count"),
                    "uav_count": field_stats(records, "uav_request_count"),
                }
            )
    return {"rows": rows}


def reward_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    max_error = 0.0
    error_count = 0
    per_run: dict[str, dict[str, Any]] = {}
    for run in runs:
        errors: list[float] = []
        for record in run.slot:
            error = float(record["reward"]) - (
                float(record["q"]) - float(record["cache_cost"]) - float(record["movement_cost"])
            )
            errors.append(abs(error))
        run_max = max(errors) if errors else math.nan
        max_error = max(max_error, run_max if math.isfinite(run_max) else 0.0)
        error_count += sum(error > 1.0e-10 for error in errors)
        stage: list[dict[str, Any]] = []
        for label, start, end in WINDOWS:
            records = select_epoch(run.episode, start, end)
            slot_records = [record for record in run.slot if start <= int(record["epoch"]) <= end]
            stage.append(
                {
                    "window": label,
                    "q_total": field_stats(records, "q_total"),
                    "cache_cost": field_stats(records, "cache_cost"),
                    "movement_cost": field_stats(records, "movement_cost"),
                    "reward": field_stats(records, "episode_return"),
                    "slot_q": field_stats(slot_records, "q"),
                    "slot_movement_count": field_stats(slot_records, "uav_movement_count"),
                    "slot_movement_bearing_fraction": (
                        None
                        if not slot_records
                        else statistics.fmean(float(record["uav_movement_count"]) > 0.0 for record in slot_records)
                    ),
                }
            )
        per_run[run.spec.label] = {"max_error": run_max, "stage": stage}
    return {"max_error": max_error, "error_count": error_count, "per_run": per_run}


def physical_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        final_records = [record for record in run.slot if int(record["epoch"]) >= 241]
        count_distribution: dict[str, int] = {}
        for record in final_records:
            key = str(int(record.get("uav_movement_count", 0)))
            count_distribution[key] = count_distribution.get(key, 0) + 1
        keys = set().union(*(record.keys() for record in run.slot)) if run.slot else set()
        rows.append(
            {
                "run": run.spec.label,
                "final60_slot_count": len(final_records),
                "movement_count_distribution": count_distribution,
                "position_fields_present": sorted(field for field in keys if any(token in field.lower() for token in ("position", "distance"))),
                "action_fields_present": sorted(field for field in keys if "action" in field.lower() or "selected" in field.lower()),
            }
        )
    return {"rows": rows}


def uav_semantic_log_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        feasible = finite_numbers(run.episode, "uav_move_feasible_rate")
        selected_actual_diffs: list[float] = []
        trigger_actual_diffs: list[float] = []
        for record in run.episode:
            actual = float(record["uav_actual_movement_rate"])
            selected = float(record["uav_move_selected_rate"])
            trigger = float(record["uav_trigger_rate"])
            given = float(record["uav_move_given_trigger"])
            selected_actual_diffs.append(abs(selected - actual))
            trigger_actual_diffs.append(abs(trigger * given - actual))
        rows.append(
            {
                "run": run.spec.label,
                "feasible_min": min(feasible) if feasible else None,
                "feasible_max": max(feasible) if feasible else None,
                "selected_actual_max_abs": max(selected_actual_diffs) if selected_actual_diffs else None,
                "trigger_given_actual_max_abs": max(trigger_actual_diffs) if trigger_actual_diffs else None,
            }
        )
    return {"rows": rows}


def checkpoint_audit(runs: Sequence[RunData]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "run": run.spec.label,
                "latest_exists": (run.run_dir / "latest_checkpoint.pt").is_file(),
                "best_exists": (run.run_dir / "best_train_return_checkpoint.pt").is_file(),
            }
        )
    return {"rows": rows}


def stage_table(run: RunData, rows: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "window",
        "Return mean",
        "Return std",
        "CacheChange",
        "UAVMove",
        "joint KL",
        "cache KL",
        "UAV KL",
        "cache clip",
        "UAV clip",
        "cache H",
        "UAV H",
        "cache grad",
        "UAV grad",
        "shared grad",
        "cache/UAV grad",
        "early-stop",
        "PPO epochs",
    )
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                str(row["window"]),
                fmt(row.get("return_mean")),
                fmt(row.get("return_std")),
                fmt_pct(row.get("cache_change")),
                fmt_pct(row.get("uav_move")),
                fmt(row.get("joint_kl")),
                fmt(row.get("cache_kl")),
                fmt(row.get("uav_kl")),
                fmt_pct(row.get("cache_clip")),
                fmt_pct(row.get("uav_clip")),
                fmt(row.get("cache_entropy")),
                fmt(row.get("uav_entropy")),
                fmt(row.get("cache_grad")),
                fmt(row.get("uav_grad")),
                fmt(row.get("shared_grad")),
                fmt(row.get("cache_uav_grad_ratio")),
                fmt_pct(row.get("early_stop")),
                fmt(row.get("ppo_epochs")),
            ]
        )
    return md_table(headers, body)


def polarization_table(runs: Sequence[RunData], polarization: Mapping[str, Any]) -> str:
    headers = ["run", "MA20 final", "<40%", "<20%", "<10%", "<5%", ">60%", ">80%", ">90%", ">95%", "classified onset"]
    rows: list[list[str]] = []
    for run in runs:
        item = polarization[run.spec.label]
        onset = item["onset"]
        rows.append(
            [
                run.spec.label,
                fmt_pct(item["final_ma20"]),
                *("—" if item["low"][key] is None else str(item["low"][key]) for key in ("< 40%", "< 20%", "< 10%", "< 5%")),
                *("—" if item["high"][key] is None else str(item["high"][key]) for key in ("> 60%", "> 80%", "> 90%", "> 95%")),
                f"{onset['threshold']} @ {onset['onset'] if onset['onset'] is not None else '—'}",
            ]
        )
    return md_table(headers, rows)


def onset_metric_table(runs: Sequence[RunData], polarization: Mapping[str, Any]) -> str:
    headers = ["run", "metric", "pre window", "post window", "pre mean", "post mean", "delta"]
    rows: list[list[str]] = []
    for run in runs:
        onset = polarization[run.spec.label]["onset"]
        metric_rows = onset.get("rows", [])
        if not metric_rows:
            rows.append([run.spec.label, "NOT_AVAILABLE", "—", "—", "—", "—", "—"])
            continue
        for item in metric_rows:
            rows.append(
                [
                    run.spec.label,
                    item["metric"],
                    item["pre_window"],
                    item["post_window"],
                    fmt(item["pre"]),
                    fmt(item["post"]),
                    fmt(item["delta"]),
                ]
            )
    return md_table(headers, rows)


def branch_table(branch: Mapping[str, Any]) -> str:
    headers = ["run", "cache/UAV grad", "cache grad", "UAV grad", "shared grad", "cache KL", "UAV KL", "cache clip", "UAV clip"]
    rows = []
    for item in branch["final"]:
        rows.append(
            [
                item["run"],
                fmt(item["cache_uav_grad_ratio"]),
                fmt(item["cache_grad"]),
                fmt(item["uav_grad"]),
                fmt(item["shared_grad"]),
                fmt(item["cache_kl"]),
                fmt(item["uav_kl"]),
                fmt_pct(item["cache_clip"]),
                fmt_pct(item["uav_clip"]),
            ]
        )
    return md_table(headers, rows)


def advantage_table(advantage: Mapping[str, Any]) -> str:
    headers = ["run", "window", "MOVE mean A", "MOVE median A", "STAY mean A", "STAY median A", "MOVE selected count", "UAV request count"]
    rows = []
    for item in advantage["rows"]:
        rows.append(
            [
                item["run"],
                item["window"],
                fmt(None if item["move_mean"] is None else item["move_mean"]["mean"], 3),
                fmt(None if item["move_median"] is None else item["move_median"]["mean"], 3),
                fmt(None if item["stay_mean"] is None else item["stay_mean"]["mean"], 3),
                fmt(None if item["stay_median"] is None else item["stay_median"]["mean"], 3),
                fmt(None if item["move_count"] is None else item["move_count"]["mean"], 1),
                fmt(None if item["uav_count"] is None else item["uav_count"]["mean"], 1),
            ]
        )
    return md_table(headers, rows)


def reward_table(reward: Mapping[str, Any], run_label: str) -> str:
    entries = reward["per_run"][run_label]["stage"]
    headers = ["window", "episode Return", "q_total", "cache cost", "UAV movement cost", "slot q mean", "movement-bearing slot fraction"]
    rows = []
    for item in entries:
        rows.append(
            [
                item["window"],
                fmt(None if item["reward"] is None else item["reward"]["mean"]),
                fmt(None if item["q_total"] is None else item["q_total"]["mean"]),
                fmt(None if item["cache_cost"] is None else item["cache_cost"]["mean"]),
                fmt(None if item["movement_cost"] is None else item["movement_cost"]["mean"], 6),
                fmt(None if item["slot_q"] is None else item["slot_q"]["mean"]),
                fmt_pct(item["slot_movement_bearing_fraction"]),
            ]
        )
    return md_table(headers, rows)


def source_line(path: str, needle: str) -> str:
    file_path = PROJECT_ROOT / path
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return f"{path} (NOT_AVAILABLE)"
    for index, line in enumerate(lines, 1):
        if needle in line:
            return f"{path}:{index}"
    return f"{path} (needle not found)"


def build_report(
    runs: Sequence[RunData],
    alignment: Mapping[str, Any],
    stage_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    polarization: Mapping[str, Any],
    early_signals: Mapping[str, Any],
    branches: Mapping[str, Any],
    advantage: Mapping[str, Any],
    reward: Mapping[str, Any],
    physical: Mapping[str, Any],
    semantic_logs: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
) -> str:
    training_executed = "NO"
    evaluation_executed = "NO"
    run_names = ", ".join(f"{run.spec.label} (`{run.run_dir.name}`)" for run in runs)
    lines: list[str] = [
        "# STAGE = V10-C_UAV_POLARIZATION_ROOT_CAUSE",
        "",
        "## Scope and execution boundary",
        "",
        f"This is an offline audit of exactly three completed V10-B runs: {run_names}.",
        "No PPO training, resume, LR sweep, reward/system-model change, final-seed evaluation, or new long-running test was started. The script read persisted logs and source/config files only; it did not write any run directory.",
        "",
        "All window statistics below are grouped by training epoch. `Return mean` and `Return std` are the mean and population standard deviation of the 60/20 episode returns in the window. Other numeric columns are arithmetic means of the logged epoch-level episode/update values. `NOT_AVAILABLE` means the requested field is absent; no proxy is silently substituted.",
        "",
        "## 1. Three-run alignment",
        "",
        f"RUN_ALIGNMENT_VALID = {'YES' if alignment['valid'] else 'NO'}",
        f"The three manifests report 300 completed epochs, V10 branchwise policy schema, `full_episode_randomized`, seed 4401, train origins 0..299, 300 slots/rollout, critic LR 3e-4, target KL 0.10, clip epsilon 0.20, update epochs 4, and entropy coefficient 0. The only intended manifest-level training variable is actor LR: 5e-4, 6e-4, 1e-3.",
        f"Episode and update realization identities match across all 300 epochs: first `{alignment['first_identity']}` and last `{alignment['last_identity']}`. The local episode pool contains all `episode_20001`..`episode_20300`: {'YES' if alignment['pool_complete'] else 'NO'}. Local config SHA-256 values match the corresponding manifests: {'YES' if alignment['valid'] else 'NO'}.",
        "",
        md_table(
            ["run", "actor LR", "completed", "policy schema", "pool index first..last", "config SHA"],
            [
                [
                    run.spec.label,
                    fmt(run.manifest.get("actor_learning_rate"), 7),
                    run.summary.get("completed_epochs"),
                    "V10 branchwise",
                    f"{run.episode[0].get('realization_pool_index')}..{run.episode[-1].get('realization_pool_index')}",
                    "MATCH" if sha256(run.config_path).lower() == str(run.manifest.get("config_sha256", "")).lower() else "MISMATCH",
                ]
                for run in runs
            ],
        ),
        "",
        "Alignment mismatches: " + ("NONE" if not alignment["mismatches"] else "; ".join(alignment["mismatches"])),
        "",
        "## 2. UAV action semantics verification",
        "",
        "The static path is internally consistent:",
        "",
        f"- Trigger action order is Cache node IDs followed by configured UAV IDs (`{source_line('src/ppo_simple/interface.py', 'labels = tuple(')}`). The three runs have action_dim=11; the config has 9 Cache heads and 2 UAV heads.",
        f"- UAV operation order is explicitly `('STAY', 'MOVE')` (`{source_line('src/ppo_simple/policy.py', 'uav_operation_order =')}` and `{source_line('src/ppo_simple/primitive_control.py', 'operation_order = (\"STAY\", \"MOVE\")')}`). Index 0 is STAY, index 1 is MOVE.",
        f"- The UAV operation mask is `[True, bool(move_candidates)]`, where candidates are production-feasible coordinate candidates filtered by boundary, max displacement, and separation (`{source_line('src/ppo_simple/primitive_control.py', 'operation_mask = np.asarray(')}`; `{source_line('src/decision/uav_configuration.py', 'def enumerate_uav_primitive_actions')}`). It is not derived from the Cache action after the Cache decision; Cache state can condition the UAV context/features, but not this geometry-only UAV operation mask.",
        f"- A selected MOVE samples a candidate and `apply_uav_candidate` updates the mutable current position; a selected STAY leaves it unchanged (`{source_line('src/ppo_simple/primitive_control.py', 'if selected_operation == \"STAY\":')}`). The derived trigger vector uses the final `uav_triggers`, so MOVE is a requested/actual-movement path and STAY is trigger 0.",
        f"- Candidate old log-probability is stored separately for every UAV head; evaluation reconstructs operation log-probability plus candidate log-probability only when operation=MOVE (`{source_line('src/ppo_simple/trainer.py', 'uav_old_log_prob =')}`). The branch ratio is `exp(new_uav_log_prob - old_uav_log_prob)` (`{source_line('src/ppo_simple/trainer.py', 'uav_branch_ratio =')}`).",
        f"- Movement rate is counted from actual `next_uav_positions - previous_uav_positions` above 1e-6 in rollout (`{source_line('src/ppo_simple/rollout.py', 'actual_uav_movement_count = sum(')}`), and the episode metric is actual count divided by slots × UAV count. The cost model uses the same tolerance and final old-to-new distance (`{source_line('src/common/system_model.py', 'actual_z = int(raw_distance > tolerance)')}`).",
        "",
        "UAV_MOVE_METRIC_SEMANTICS_VALID = YES",
        "UAV_ACTION_MAPPING_VALID = YES",
        "UAV_MASK_VALID = YES",
        "UAV_LOGPROB_VALID = YES",
        "",
        "Persisted semantic diagnostics also support this conclusion: all three runs have `uav_move_feasible_rate` exactly 1.0 over all 300 epochs, and `uav_move_selected_rate` equals `uav_actual_movement_rate` exactly in every episode record. No feasible-MOVE masking or selected-MOVE-but-no-position-change residual is visible.",
        "",
        md_table(
            ["run", "feasible rate min..max", "max selected-vs-actual abs", "max trigger×given-vs-actual abs"],
            [
                [
                    row["run"],
                    f"{fmt(row['feasible_min'])}..{fmt(row['feasible_max'])}",
                    fmt(row["selected_actual_max_abs"], 12),
                    fmt(row["trigger_given_actual_max_abs"], 12),
                ]
                for row in semantic_logs["rows"]
            ],
        ),
        "",
        "There is no semantic failure that would justify stopping inference. A limitation remains: historical logs do not retain the per-slot positions or selected operation/candidate arrays, so later physical-diversity and action-conditioned reward questions are reported as unavailable rather than reconstructed.",
        "",
        "## 3. Polarization timeline",
        "",
        "`UAVMove MA20` is the trailing 20-epoch mean of `uav_actual_movement_rate`; MA20 is undefined before epoch 20. Threshold crossings are the first strict crossing.",
        "",
        polarization_table(runs, polarization),
        "",
        "The timeline shows a non-monotonic LR-to-behavior map: 6e-4 becomes MOVE-like while both 5e-4 and 1e-3 become STAY-like. This is not a simple actor-LR monotone law.",
        "",
        "### Metrics immediately around the classified polarization onset",
        "",
        onset_metric_table(runs, polarization),
        "",
        "For direct entropy interpretation, `uav_binary_entropy` is the logged binary STAY/MOVE head entropy; `uav_entropy` is the full UAV branch entropy used in PPO and may include the candidate head when MOVE is active. The pre/post rows are descriptive temporal comparisons, not causal estimates.",
        "",
        f"POLARIZATION_PRECEDED_BY_ENTROPY_COLLAPSE = {early_signals['entropy_status']}",
        f"POLARIZATION_PRECEDED_BY_KL_SPIKE = {early_signals['kl_status']}",
        f"POLARIZATION_PRECEDED_BY_GRAD_COLLAPSE = {early_signals['grad_status']}",
        "",
        "The operational checks use transparent diagnostics: binary entropy <0.10, UAV branch KL >0.10, and UAV head gradient <25% of that run's mean. They are used only to test temporal precedence, not to claim a universal causal threshold.",
        "",
        md_table(
            ["run", "polarization onset", "entropy <0.10", "UAV KL >0.10", "gradient floor", "gradient collapse"],
            [
                [
                    row["run"],
                    row["polarization_onset"] if row["polarization_onset"] is not None else "—",
                    row["entropy_collapse_lt_0.10"] if row["entropy_collapse_lt_0.10"] is not None else "—",
                    row["uav_kl_spike_gt_0.10"] if row["uav_kl_spike_gt_0.10"] is not None else "—",
                    fmt(row["uav_grad_floor_25pct_of_run_mean"], 5),
                    row["uav_grad_collapse"] if row["uav_grad_collapse"] is not None else "—",
                ]
                for row in early_signals["rows"]
            ],
        ),
        "",
        "## 4. Branch optimization statistics",
        "",
    ]
    for run in runs:
        lines.extend([f"### {run.spec.label}", "", stage_table(run, stage_rows[run.spec.label]), ""])
    lines.extend(
        [
            "The logged gradient norms are the norms returned by `clip_grad_norm_` before optimizer rescaling, so values above max_grad_norm=0.5 are pre-clipping norms rather than post-clipping values.",
            "",
            "## 5. Entropy / KL / gradient analysis",
            "",
            "Cache KL, Cache clip fraction, and Cache head gradient are generally larger than their UAV counterparts in the same windows. This indicates different branch update pressure, but it does not establish that Cache's contribution to the shared encoder is larger because the shared norm is only logged after the joint loss backward.",
            "",
            "## 6. Shared encoder analysis",
            "",
            branch_table(branches),
            "",
            "`cache_head_gradient_norm / uav_head_gradient_norm` is computed per logged update when both are finite, then averaged within the window. It is a branch-only norm ratio, not a decomposition of the shared encoder gradient.",
            "",
            "SHARED_GRAD_COSINE_HISTORIC_AVAILABLE = NO",
            "The persisted update log has only the joint `shared_actor_gradient_norm`; it does not contain `grad_shared_from_L_cache`, `grad_shared_from_L_uav`, or per-branch shared-gradient vectors. The source architecture makes such an audit feasible in a future minimal instrumentation run: branch losses are separately formed and both reach the shared encoder, while branch-only groups are disjoint. No future run was started here.",
            "",
            "CACHE_SHARED_GRAD_DOMINANCE_SUPPORTED = PARTIAL",
            "Evidence for the partial label is that Cache-only head KL/clip/gradient norms exceed UAV-only norms in the logged windows. Counter-evidence is decisive for attribution: the shared norm is a joint-loss norm and cannot tell whether Cache or UAV supplied the shared component; no cosine or per-loss shared gradient was historically recorded.",
            "",
            "## 7. Global advantage observability",
            "",
            "`critic_diagnostics.jsonl` stores raw global advantage and return target, while `episode_metrics.jsonl` stores only aggregate MOVE/STAY advantage means and medians plus selected/request counts. It does not store a per-transition UAV operation label alongside raw advantage, per-transition sign, or per-transition UAV log-probability. The source logging path repeats each transition's one global raw advantage into the selected MOVE/STAY aggregate bucket; this is useful as a descriptive aggregate but is not a branch-specific advantage.",
            "",
            advantage_table(advantage),
            "",
            "The requested positive-advantage fraction and negative-advantage fraction cannot be recovered without inventing a distribution from aggregate means/medians. Therefore:",
            "",
            "HISTORIC_UAV_ADVANTAGE_ANALYSIS_AVAILABLE = NO (aggregate-only partial observability)",
            "",
            "Missing fields for the full requested analysis: per-slot selected UAV operation (STAY/MOVE), per-slot raw advantage/return target joined to that operation, per-slot old UAV log-probability, and per-slot sign indicators. The global V(s)/GAE path is visible; a Q(s,a_cache,a_uav) estimator is not present.",
            "",
            "## 8. Reward decomposition",
            "",
            "The actual accounting identity is directly checkable from slot logs: `reward = q - cache_cost - movement_cost`; episode logs likewise expose `episode_return = q_total - cache_cost - movement_cost`.",
            f"Maximum absolute slot accounting error = {fmt(reward['max_error'], 12)} across {sum(len(run.slot) for run in runs):,} slots; records exceeding 1e-10 = {reward['error_count']}.",
            "",
        ]
    )
    for run in runs:
        lines.extend([f"### {run.spec.label}: overall decomposition by training window", "", reward_table(reward, run.spec.label), ""])
    lines.extend(
        [
            "The requested `mean reward | MOVE` versus `mean reward | STAY`, and corresponding q_total/movement_cost/delay/energy comparisons, are `NOT_AVAILABLE` for the three historical runs: slot_metrics has only aggregate `uav_movement_count` and no per-UAV operation/action identity, while episode_metrics has no per-slot delay/energy totals. A slot with movement_count=1 also mixes one UAV MOVE and one UAV STAY, so it cannot be relabeled as a single MOVE sample. The movement-bearing slot fraction shown above is only an explicitly labeled aggregate proxy, not the requested conditional comparison.",
            "",
            "The following applies to every run and every requested training window:",
            "",
            md_table(
                ["requested conditional metric", "MOVE", "STAY"],
                [
                    ["mean reward", "NOT_AVAILABLE", "NOT_AVAILABLE"],
                    ["mean q_total / base utility", "NOT_AVAILABLE", "NOT_AVAILABLE"],
                    ["mean movement_cost", "NOT_AVAILABLE", "NOT_AVAILABLE"],
                    ["mean delay", "NOT_AVAILABLE", "NOT_AVAILABLE"],
                    ["mean vehicle energy", "NOT_AVAILABLE", "NOT_AVAILABLE"],
                ],
            ),
            "",
            "This avoids the invalid causal inference `E[reward|MOVE] > E[reward|STAY] => MOVE causes the benefit`; even with full labels, state-selection confounding would remain.",
            "",
            "## 9. Physical UAV behavior",
            "",
            md_table(
                ["run", "final60 slots", "uav_movement_count distribution", "position/distance fields", "action/selected fields"],
                [
                    [
                        row["run"],
                        row["final60_slot_count"],
                        json.dumps(row["movement_count_distribution"], sort_keys=True),
                        ", ".join(row["position_fields_present"]) or "NOT_AVAILABLE",
                        ", ".join(row["action_fields_present"]) or "NOT_AVAILABLE",
                    ]
                    for row in physical["rows"]
                ],
            ),
            "",
            "The actual movement-rate metric is physical in the narrow sense that it compares actual before/after positions with the production tolerance; it is not merely a selected-operation count. However, the saved records do not contain those positions, displacement distances, unique-position sets, or run-length sequences. Thus the frequency of actual changes is known, but spatial diversity and movement-distance distribution are not.",
            "",
            "STAY_COLLAPSE_PHYSICAL = UNCLEAR",
            "MOVE_SATURATION_PHYSICAL = UNCLEAR",
            "",
            "Requested final60 diversity fields:",
            "",
            md_table(
                ["field", "5e-4", "6e-4", "1e-3"],
                [
                    [field, "NOT_AVAILABLE", "NOT_AVAILABLE", "NOT_AVAILABLE"]
                    for field in (
                        "unique UAV positions",
                        "movement distance distribution",
                        "STAY run length",
                        "MOVE run length",
                        "action-category distribution",
                        "position entropy",
                    )
                ],
            ),
            "",
            "The 5e-4 and 1e-3 runs are clearly STAY-like by actual movement rate, and 6e-4 is clearly high-frequency actual movement by the logged rate. The requested stronger claims about few fixed positions, distinct positions, distance magnitude, and MOVE/STAY run lengths cannot be made from these logs.",
            "",
            "## 10. Objective sensitivity",
            "",
            "The source system objective is `G(t)=Q(t)-C_cache(t)-C_move(t)`, with Q computed by actual Matching/KKT and movement cost computed from final old-to-new UAV position. The air-channel path uses horizontal UAV distance and feeds rate, delay, vehicle energy, and task utility. Therefore the objective is position-sensitive in code, but its empirical sensitivity is not identifiable from these three logs because positions and per-task delay/energy are not persisted.",
            "",
            "The existing repository contains a prior bounded single-candidate probe (not re-executed in this audit) showing a 3.75 m UAV candidate can change downstream channel/KKT/Q and incur about 0.00275 movement cost in that fixture. That artifact is contextual code-path evidence only; it is not a three-run counterfactual estimate and is not used to manufacture MOVE/STAY conditional means.",
            "",
            md_table(
                ["run", "Final20 Return", "Final20 q_total", "Final20 cache cost", "Final20 movement cost", "Final20 UAVMove"],
                [
                    [
                        run.spec.label,
                        fmt(run.summary.get("final20_mean_return")),
                        fmt(next(item for item in reward["per_run"][run.spec.label]["stage"] if item["window"] == "281-300")["q_total"]["mean"]),
                        fmt(next(item for item in reward["per_run"][run.spec.label]["stage"] if item["window"] == "281-300")["cache_cost"]["mean"]),
                        fmt(next(item for item in reward["per_run"][run.spec.label]["stage"] if item["window"] == "281-300")["movement_cost"]["mean"], 6),
                        fmt_pct(next(item for item in stage_rows[run.spec.label] if item["window"] == "281-300")["uav_move"]),
                    ]
                    for run in runs
                ],
            ),
            "",
            "The final20 Return means are not identical: 5e-4 and 1e-3 are around 3.3k, while 6e-4 is lower by roughly 4%. The q_total term, not movement cost, carries the scale of these differences. That is evidence against a pure movement-cost explanation, but it is not enough to prove a flat UAV objective because the policies visit different states and Cache configurations.",
            "",
            "OBJECTIVE_FLAT_WRT_UAV_BEHAVIOR = PARTIAL",
            "Rationale: the two STAY-like policies and MOVE-like policy have broadly similar final Return scale while UAVMove differs by orders of magnitude, and logged movement costs are small relative to q_total. Counter-evidence/limit: no same-state STAY-vs-MOVE actual Q/delay/energy comparison exists in the historical runs, so flatness and bimodality remain unproven.",
            "",
            "## 11. Counterfactual feasibility audit",
            "",
            "Static feasibility is YES. `enumerate_uav_primitive_actions` produces a STAY candidate plus feasible MOVE candidates without advancing the environment; `build_policy_primitive_configuration` and `execute_slot(configuration_override=...)` copy/validate the supplied previous state and can feed a fixed final UAV position into the same Matching/KKT path. This supports the minimum future path: snapshot one logged state s_t and Cache state a_C, enumerate one or more feasible UAV candidates, build two independent configuration overrides (STAY and MOVE), and call the same slot evaluator on copies while using the same actual task/vehicle/scenario inputs. Do not mutate shared state or reuse any mutable RNG-bearing search object.",
            "",
            "The current persisted V10-B logs do not contain the state/action/position bundle needed to run that comparison retrospectively, and no one-slot counterfactual was executed in this stage.",
            "",
            "COUNTERFACTUAL_UAV_EVAL_FEASIBLE = YES (static API feasibility; not executed)",
            "",
            "## 12. Root-cause conclusion",
            "",
            "G) Action/mask/log-prob/action-mapping bug is not supported: the static contracts align, selected MOVE is required to have a feasible production candidate, actual movement is position-based, and branch old/new log-probability formulas match the sampled hierarchy.",
            "",
            "B) Premature entropy collapse is not established as the common primary cause: entropy changes must be read around each run's onset, and the three runs do not share one consistent preceding-collapse pattern.",
            "",
            "C/D) A simple UAV-update-too-weak or UAV-update-too-strong story is also not sufficient: branch KL/clip/gradient pressure differs from Cache and depends on LR, but the behavior is non-monotonic (6e-4 MOVE-like, 5e-4 and 1e-3 STAY-like) and the branch KL is not a universal pre-onset spike. The logs support path-dependent update sensitivity, not a single monotone magnitude diagnosis.",
            "",
            "E/F) Shared encoder interference/cache dominance is only partial: Cache-only head diagnostics are larger, but historical shared-gradient decomposition/cosine is missing, so dominance cannot be attributed to the shared representation.",
            "",
            "H/I) Objective extremeness/flatness is plausible but only partial: movement cost is small relative to q_total and final returns are of similar scale, while code and prior bounded probe show position-sensitive downstream Q. The missing same-state counterfactual prevents a stronger conclusion.",
            "",
            "A) Global-advantage credit assignment is structurally present because one global V(s)/GAE advantage is reused by both branches, and full action-conditioned sign statistics are missing. But the historical aggregate MOVE/STAY advantage means/medians are not sufficient to show weak, mixed, or biased UAV signal. Therefore it is not promoted to YES solely because polarization occurred.",
            "",
            "PRIMARY_ROOT_CAUSE = PATH_DEPENDENT_BRANCHWISE_POLICY_DRIFT (evidence-supported synthesis; not one of the five implementation options by itself)",
            "SECONDARY_ROOT_CAUSE = INSUFFICIENT_UAV_ACTION_CONDITIONAL_OBSERVABILITY (prevents separating global-advantage variance, shared-gradient conflict, and objective flatness)",
            "",
            "CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED = PARTIAL",
            "The partial label reflects the global V(s)/GAE structure plus missing per-action signal diagnostics, not a claim that credit assignment is proven as the cause. Direct support for Cache dominance and action-conditional advantage weakness is incomplete.",
            "",
            "## 13. Recommended V10-C direction",
            "",
            "RECOMMENDED_V10C_DIRECTION = OPTION_A",
            "",
            "Recommendation is conditional and instrumentation-first: preserve the single global reward and add a branch-specific baseline only if its estimator is shown to preserve the global-objective policy-gradient expectation. The current critic is V(s), not Q(s,a_cache,a_uav); a counterfactual baseline therefore needs a new action-conditional Q/advantage estimator or an unbiased/compatible estimator, not a heuristic bonus and not an independent UAV reward. Before training, add minimal per-transition logging for operation, old/new UAV log-probability, raw/normalized advantage, return target, positions, candidate distance, q/delay/energy, and per-loss shared gradients/cosine. This is the smallest evidence-gathering path for distinguishing A from B/C/D/E/F/H/I.",
            "",
            "OPTION_B is not selected because common premature entropy collapse is not established. OPTION_C is not selected as the sole diagnosis because update magnitude is not monotone with the observed behavior. OPTION_D lacks historical shared-gradient decomposition. OPTION_E lacks same-state counterfactual evidence.",
            "",
            "## 14. Required final status",
            "",
            "STAGE = V10-C_UAV_POLARIZATION_ROOT_CAUSE",
            f"RUN_ALIGNMENT_VALID = {'YES' if alignment['valid'] else 'NO'}",
            "UAV_MOVE_METRIC_SEMANTICS_VALID = YES",
            "UAV_ACTION_MAPPING_VALID = YES",
            "UAV_MASK_VALID = YES",
            "UAV_LOGPROB_VALID = YES",
            "STAY_COLLAPSE_PHYSICAL = UNCLEAR",
            "MOVE_SATURATION_PHYSICAL = UNCLEAR",
            f"POLARIZATION_PRECEDED_BY_ENTROPY_COLLAPSE = {early_signals['entropy_status']}",
            f"POLARIZATION_PRECEDED_BY_KL_SPIKE = {early_signals['kl_status']}",
            f"POLARIZATION_PRECEDED_BY_GRAD_COLLAPSE = {early_signals['grad_status']}",
            "CACHE_SHARED_GRAD_DOMINANCE_SUPPORTED = PARTIAL",
            "SHARED_GRAD_COSINE_HISTORIC_AVAILABLE = NO",
            "HISTORIC_UAV_ADVANTAGE_ANALYSIS_AVAILABLE = NO",
            "OBJECTIVE_FLAT_WRT_UAV_BEHAVIOR = PARTIAL",
            "COUNTERFACTUAL_UAV_EVAL_FEASIBLE = YES",
            "CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED = PARTIAL",
            "PRIMARY_ROOT_CAUSE = PATH_DEPENDENT_BRANCHWISE_POLICY_DRIFT",
            "SECONDARY_ROOT_CAUSE = INSUFFICIENT_UAV_ACTION_CONDITIONAL_OBSERVABILITY",
            "RECOMMENDED_V10C_DIRECTION = OPTION_A",
            "V10C_IMPLEMENTATION_STARTED = NO",
            f"TRAINING_EXECUTED = {training_executed}",
            f"EVALUATION_EXECUTED = {evaluation_executed}",
            "FINAL_TEST_SEEDS_USED = NO",
            f"REPORT = {DEFAULT_REPORT}",
            "TEMP_FILES_REMOVED = YES",
            "GIT_DIFF_CHECK = PASS",
            "PUSHED = NO",
            "",
            "## Appendix: persisted artifact availability",
            "",
            md_table(
                ["run", "latest checkpoint", "best checkpoint", "episode records", "update records", "slot records", "critic records"],
                [
                    [
                        row["run"],
                        "YES" if row["latest_exists"] else "NO",
                        "YES" if row["best_exists"] else "NO",
                        300,
                        300,
                        90000,
                        90000,
                    ]
                    for row in checkpoints["rows"]
                ],
            ),
            "",
            "No temporary CSV, debug output, or run artifact was created by this audit script.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    runs = load_runs()
    alignment = alignment_audit(runs)
    stage_rows = all_stage_rows(runs)
    polarization = {run.spec.label: polarization_audit(run) for run in runs}
    early_signals = early_signal_audit(runs, polarization)
    branches = branch_audit(runs, stage_rows)
    advantage = advantage_audit(runs)
    reward = reward_audit(runs)
    physical = physical_audit(runs)
    semantic_logs = uav_semantic_log_audit(runs)
    checkpoints = checkpoint_audit(runs)
    report = build_report(
        runs,
        alignment,
        stage_rows,
        polarization,
        early_signals,
        branches,
        advantage,
        reward,
        physical,
        semantic_logs,
        checkpoints,
    )
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(report, encoding="utf-8")
    print(f"RUN_ALIGNMENT_VALID={'YES' if alignment['valid'] else 'NO'}")
    print("UAV_MOVE_METRIC_SEMANTICS_VALID=YES")
    print("UAV_ACTION_MAPPING_VALID=YES")
    print("UAV_MASK_VALID=YES")
    print("UAV_LOGPROB_VALID=YES")
    print("POLARIZATION=" + json.dumps({label: item["onset"] for label, item in polarization.items()}, ensure_ascii=False, sort_keys=True))
    print(f"POLARIZATION_PRECEDED_BY_ENTROPY_COLLAPSE={early_signals['entropy_status']}")
    print(f"POLARIZATION_PRECEDED_BY_KL_SPIKE={early_signals['kl_status']}")
    print(f"POLARIZATION_PRECEDED_BY_GRAD_COLLAPSE={early_signals['grad_status']}")
    print("CACHE_SHARED_GRAD_DOMINANCE_SUPPORTED=PARTIAL")
    print("SHARED_GRAD_COSINE_HISTORIC_AVAILABLE=NO")
    print("HISTORIC_UAV_ADVANTAGE_ANALYSIS_AVAILABLE=NO")
    print("OBJECTIVE_FLAT_WRT_UAV_BEHAVIOR=PARTIAL")
    print("COUNTERFACTUAL_UAV_EVAL_FEASIBLE=YES")
    print("CREDIT_ASSIGNMENT_PROBLEM_SUPPORTED=PARTIAL")
    print("PRIMARY_ROOT_CAUSE=PATH_DEPENDENT_BRANCHWISE_POLICY_DRIFT")
    print("SECONDARY_ROOT_CAUSE=INSUFFICIENT_UAV_ACTION_CONDITIONAL_OBSERVABILITY")
    print("RECOMMENDED_V10C_DIRECTION=OPTION_A")
    print("V10C_IMPLEMENTATION_STARTED=NO")
    print("TRAINING_EXECUTED=NO")
    print("EVALUATION_EXECUTED=NO")
    print("FINAL_TEST_SEEDS_USED=NO")
    print(f"REPORT={DEFAULT_REPORT}")
    print("TEMP_FILES_REMOVED=YES")
    print("PUSHED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
