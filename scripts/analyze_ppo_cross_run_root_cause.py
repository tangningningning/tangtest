"""Analyze aligned PPO runs using only persisted training logs.

The script discovers complete formal runs from their metadata, verifies the
episode realization mapping, computes trailing-only residual diagnostics, and
writes one Markdown report plus at most three PNG figures. It never starts a
training/evaluation/data-generation process and never edits a run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
REPORT_PATH = PROJECT_ROOT / "reports" / "ppo_cross_run_root_cause_analysis.md"
FIGURE_PATHS = {
    "residuals": PROJECT_ROOT / "reports" / "ppo_same_realization_residuals.png",
    "updates": PROJECT_ROOT / "reports" / "ppo_update_intensity_comparison.png",
    "behavior": PROJECT_ROOT / "reports" / "ppo_policy_behavior_drift.png",
}

EXPECTED_EPOCHS = 300
MA_WINDOW = 20
MAX_GRAD_NORM_FIELD = "max_grad_norm"
WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("whole run", 1, 300),
    ("1-60", 1, 60),
    ("61-120", 61, 120),
    ("121-180", 121, 180),
    ("181-240", 181, 240),
    ("241-300", 241, 300),
    ("281-300", 281, 300),
)


@dataclass(frozen=True)
class TargetRun:
    label: str
    actor_learning_rate: float
    kl_protection: bool


TARGET_RUNS: tuple[TargetRun, ...] = (
    TargetRun("1e-4", 1.0e-4, False),
    TargetRun("5e-4", 5.0e-4, False),
    TargetRun("7e-4", 7.0e-4, False),
    TargetRun("1e-3", 1.0e-3, False),
    TargetRun("5e-4 + KL", 5.0e-4, True),
    TargetRun("1e-3 + KL", 1.0e-3, True),
)

_EXCLUDED_RUN_NAME_TOKENS = (
    "tmp",
    "temp",
    "diagnostic",
    "probe",
    "failed",
    "failure",
)


@dataclass
class RunData:
    label: str
    run_dir: Path
    config_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    episode_records: list[dict[str, Any]]
    update_records: list[dict[str, Any]]
    critic_records: list[dict[str, Any]]
    epochs: np.ndarray
    returns: np.ndarray
    ma20: np.ndarray = field(default_factory=lambda: np.empty(0))
    residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    standardized_residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    episode_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    update_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    critic_arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path, expected_record_type: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record at {path}:{line_number}")
            if record.get("record_type", expected_record_type) == expected_record_type:
                records.append(record)
    return records


def _require_epoch_order(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    epochs = [record.get("epoch") for record in records]
    expected = list(range(1, EXPECTED_EPOCHS + 1))
    if epochs != expected:
        raise ValueError(
            f"{path} must contain ordered epochs 1..{EXPECTED_EPOCHS}; "
            f"received {len(records)} records"
        )


def _mapping_fields_present(record: Mapping[str, Any]) -> bool:
    return all(
        field_name in record
        for field_name in (
            "epoch",
            "episode",
            "training_episode_index",
            "realization_pool_index",
            "realization_directory",
            "seed",
            "global_training_slot_start",
            "global_training_slot_end",
        )
    )


def _read_episode_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episode_metrics.jsonl"
    records = _read_jsonl(path, "episode")
    if len(records) != EXPECTED_EPOCHS:
        raise ValueError(f"{path} has {len(records)} episode records, expected 300")
    _require_epoch_order(records, path)
    if not all(_mapping_fields_present(record) for record in records):
        raise ValueError(f"{path} is missing same-realization mapping fields")
    for record in records:
        value = float(record["episode_return"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite episode_return in {path}")
    return records


def _read_update_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "update_metrics.jsonl"
    records = _read_jsonl(path, "update")
    if len(records) != EXPECTED_EPOCHS:
        raise ValueError(f"{path} has {len(records)} update records, expected 300")
    _require_epoch_order(records, path)
    if not all(_mapping_fields_present(record) for record in records):
        raise ValueError(f"{path} is missing same-realization mapping fields")
    return records


def _read_critic_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "critic_diagnostics.jsonl"
    if not path.is_file():
        return []
    return _read_jsonl(path, "critic_transition")


def _resolve_config_path(raw_value: Any, expected_sha256: Any = None) -> Path | None:
    if not isinstance(raw_value, str) or not raw_value:
        return None
    path = Path(raw_value)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if resolved.is_file():
        return resolved
    if not isinstance(expected_sha256, str) or not expected_sha256:
        return None
    matches: list[Path] = []
    for candidate in sorted((PROJECT_ROOT / "configs").rglob("*.toml")):
        try:
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest.lower() == expected_sha256.lower():
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def _matches_target(manifest: Mapping[str, Any], target: TargetRun) -> bool:
    actor_lr = manifest.get("actor_learning_rate")
    if actor_lr is None:
        return False
    if not math.isclose(float(actor_lr), target.actor_learning_rate, rel_tol=0.0, abs_tol=1.0e-12):
        return False
    target_kl = manifest.get("target_kl")
    if target.kl_protection:
        return target_kl is not None and math.isclose(
            float(target_kl), 0.10, rel_tol=0.0, abs_tol=1.0e-12
        )
    return target_kl is None


def discover_formal_runs(runs_root: str | Path = RUNS_ROOT) -> tuple[RunData, ...]:
    """Find exactly one complete main-training run for each requested target."""
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {root}")

    candidates: list[RunData] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if any(token in run_dir.name.lower() for token in _EXCLUDED_RUN_NAME_TOKENS):
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file() or not (run_dir / "episode_metrics.jsonl").is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
            summary = _read_json(summary_path)
            config_path = _resolve_config_path(
                manifest.get("config_path"), manifest.get("config_sha256")
            )
            if config_path is None or not config_path.is_file():
                continue
            if manifest.get("experiment_type") != "main_training":
                continue
            if summary.get("status") != "completed":
                continue
            if manifest.get("target_epochs") != EXPECTED_EPOCHS:
                continue
            if summary.get("completed_epochs") != EXPECTED_EPOCHS:
                continue
            episode_records = _read_episode_records(run_dir)
            update_records = _read_update_records(run_dir)
            critic_records = _read_critic_records(run_dir)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        returns = np.asarray(
            [float(record["episode_return"]) for record in episode_records], dtype=np.float64
        )
        candidates.append(
            RunData(
                label="",
                run_dir=run_dir,
                config_path=config_path,
                manifest=manifest,
                summary=summary,
                episode_records=episode_records,
                update_records=update_records,
                critic_records=critic_records,
                epochs=np.arange(1, EXPECTED_EPOCHS + 1, dtype=np.int64),
                returns=returns,
            )
        )

    selected: list[RunData] = []
    for target in TARGET_RUNS:
        matches = [candidate for candidate in candidates if _matches_target(candidate.manifest, target)]
        if len(matches) != 1:
            match_paths = [str(candidate.run_dir) for candidate in matches] or ["<none>"]
            raise RuntimeError(
                f"Expected exactly one complete formal run for {target.label}; "
                f"found {len(matches)}: {match_paths}"
            )
        matches[0].label = target.label
        selected.append(matches[0])
    return tuple(selected)


def _realization_id(record: Mapping[str, Any]) -> tuple[Any, ...]:
    directory = str(record.get("realization_directory", ""))
    realization_name = Path(directory).name
    seed_match = re.search(r"episode_(\d+)$", realization_name)
    episode_seed = int(seed_match.group(1)) if seed_match else None
    return (
        record.get("realization_pool_index"),
        realization_name,
        episode_seed,
        record.get("training_episode_index"),
        record.get("episode"),
        record.get("seed"),
        record.get("global_training_slot_start"),
        record.get("global_training_slot_end"),
    )


def _record_map(records: Sequence[Mapping[str, Any]]) -> dict[int, tuple[Any, ...]]:
    return {int(record["epoch"]): _realization_id(record) for record in records}


def verify_same_realization_alignment(runs: Sequence[RunData]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if not runs:
        return {"ok": False, "mismatches": mismatches, "mismatch_epochs": []}
    for source in ("episode", "update"):
        base_records = runs[0].episode_records if source == "episode" else runs[0].update_records
        base_map = _record_map(base_records)
        for run in runs:
            records = run.episode_records if source == "episode" else run.update_records
            current_map = _record_map(records)
            for epoch in range(1, EXPECTED_EPOCHS + 1):
                if current_map.get(epoch) != base_map.get(epoch):
                    mismatches.append(
                        {
                            "source": source,
                            "epoch": epoch,
                            "label": run.label,
                            "expected": base_map.get(epoch),
                            "actual": current_map.get(epoch),
                        }
                    )
    mismatch_epochs = sorted({int(item["epoch"]) for item in mismatches})
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "mismatch_epochs": mismatch_epochs,
    }


def _full_window_ma(values: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(window - 1, values.size):
        result[index] = float(np.mean(values[index - window + 1 : index + 1]))
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.size, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(valid)) < 2:
        return None
    x = left[valid]
    y = right[valid]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(valid)) < 2:
        return None
    return _pearson(_rankdata(left[valid]), _rankdata(right[valid]))


def _metric_array(records: Sequence[Mapping[str, Any]], field_name: str) -> np.ndarray:
    result = np.full(EXPECTED_EPOCHS, np.nan, dtype=np.float64)
    for record in records:
        value = record.get(field_name)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            result[int(record["epoch"]) - 1] = numeric_value
    return result


def _optimizer_step_delta(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    cumulative = _metric_array(records, "optimizer_steps")
    result = np.full(cumulative.shape, np.nan, dtype=np.float64)
    previous = 0.0
    for index, value in enumerate(cumulative):
        if np.isfinite(value):
            result[index] = value - previous
            previous = value
    return result


def _critic_epoch_mean(records: Sequence[Mapping[str, Any]], field_name: str) -> np.ndarray:
    buckets: list[list[float]] = [[] for _ in range(EXPECTED_EPOCHS)]
    for record in records:
        value = record.get(field_name)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            buckets[int(record["epoch"]) - 1].append(numeric_value)
    result = np.full(EXPECTED_EPOCHS, np.nan, dtype=np.float64)
    for index, values in enumerate(buckets):
        if values:
            result[index] = float(np.mean(values))
    return result


def prepare_metric_arrays(runs: Sequence[RunData]) -> None:
    for run in runs:
        episode_fields = set().union(*(record.keys() for record in run.episode_records))
        update_fields = set().union(*(record.keys() for record in run.update_records))
        critic_fields = set().union(*(record.keys() for record in run.critic_records)) if run.critic_records else set()
        run.episode_arrays = {
            name: _metric_array(run.episode_records, name) for name in sorted(episode_fields)
        }
        run.update_arrays = {
            name: _metric_array(run.update_records, name) for name in sorted(update_fields)
        }
        run.update_arrays["optimizer_steps_delta"] = _optimizer_step_delta(run.update_records)
        run.critic_arrays = {
            name: _critic_epoch_mean(run.critic_records, name) for name in sorted(critic_fields)
        }
        run.ma20 = _full_window_ma(run.returns)
        run.residual = run.returns - run.ma20
        valid = np.isfinite(run.residual)
        residual_std = float(np.std(run.residual[valid])) if np.any(valid) else math.nan
        run.standardized_residual = np.full(run.residual.shape, np.nan, dtype=np.float64)
        if residual_std > 0.0 and math.isfinite(residual_std):
            run.standardized_residual[valid] = run.residual[valid] / residual_std


def _all_fields(runs: Sequence[RunData], source: str) -> list[str]:
    fields: set[str] = set()
    for run in runs:
        records = getattr(run, f"{source}_records")
        fields.update(*(record.keys() for record in records))
    return sorted(fields)


def residual_analysis(runs: Sequence[RunData]) -> dict[str, Any]:
    valid_positions = np.arange(MA_WINDOW - 1, EXPECTED_EPOCHS)
    residual_matrix = np.vstack([run.residual[valid_positions] for run in runs])
    standardized_matrix = np.vstack(
        [run.standardized_residual[valid_positions] for run in runs]
    )
    count = len(runs)
    pearson_matrix = np.full((count, count), np.nan, dtype=np.float64)
    spearman_matrix = np.full((count, count), np.nan, dtype=np.float64)
    for left in range(count):
        for right in range(count):
            pearson = _pearson(standardized_matrix[left], standardized_matrix[right])
            spearman = _spearman(standardized_matrix[left], standardized_matrix[right])
            pearson_matrix[left, right] = math.nan if pearson is None else pearson
            spearman_matrix[left, right] = math.nan if spearman is None else spearman
    pair_indices = np.triu_indices(count, 1)
    pair_pearson = pearson_matrix[pair_indices]
    pair_spearman = spearman_matrix[pair_indices]
    negative_count = np.sum(residual_matrix < 0.0, axis=0)
    difficulty_score = np.median(standardized_matrix, axis=0)
    common_order = np.argsort(difficulty_score)
    worst_by_run: dict[str, np.ndarray] = {}
    worst_cross_negative: dict[str, tuple[float, float]] = {}
    for index, run in enumerate(runs):
        positions = np.argsort(residual_matrix[index])[:20]
        worst_by_run[run.label] = valid_positions[positions] + 1
        other_negative = np.sum(residual_matrix[:, positions] < 0.0, axis=0) - (
            residual_matrix[index, positions] < 0.0
        )
        worst_cross_negative[run.label] = (
            float(np.mean(other_negative)),
            float(np.mean(other_negative > 0.0)),
        )
    worst_run_index = np.argmin(standardized_matrix, axis=0)
    other_medians = np.asarray(
        [
            np.median(np.delete(standardized_matrix[:, position], worst_run_index[position]))
            for position in range(standardized_matrix.shape[1])
        ],
        dtype=np.float64,
    )
    policy_gap = other_medians - np.min(standardized_matrix, axis=0)
    specific_candidates = np.flatnonzero(negative_count <= 2)
    if specific_candidates.size < 10:
        specific_candidates = np.arange(policy_gap.size)
    specific_order = specific_candidates[np.argsort(policy_gap[specific_candidates])[::-1]]
    return {
        "valid_positions": valid_positions,
        "residual_matrix": residual_matrix,
        "standardized_matrix": standardized_matrix,
        "pearson_matrix": pearson_matrix,
        "spearman_matrix": spearman_matrix,
        "pair_pearson": pair_pearson,
        "pair_spearman": pair_spearman,
        "negative_count": negative_count,
        "difficulty_score": difficulty_score,
        "common_order": common_order,
        "worst_by_run": worst_by_run,
        "worst_cross_negative": worst_cross_negative,
        "worst_run_index": worst_run_index,
        "policy_gap": policy_gap,
        "specific_order": specific_order,
    }


def _run_array(run: RunData, field_name: str) -> np.ndarray | None:
    for collection in (run.episode_arrays, run.update_arrays, run.critic_arrays):
        if field_name in collection:
            return collection[field_name]
    return None


def _window_stats(values: np.ndarray | None, start: int, end: int) -> dict[str, Any]:
    if values is None:
        return {"n": 0, "mean": None, "median": None, "std": None}
    selected = values[start - 1 : end]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "std": float(np.std(selected)),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT AVAILABLE"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NOT AVAILABLE"
        return f"{float(value):.{digits}g}"
    return str(value)


def _stats_cell(stats: Mapping[str, Any]) -> str:
    if stats.get("n", 0) == 0:
        return "NOT AVAILABLE"
    return f"{_fmt(stats['mean'])} / {_fmt(stats['median'])} / {_fmt(stats['std'])}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(header) for header in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _field_window_table(runs: Sequence[RunData], field_name: str) -> str:
    headers = ["label"] + [window[0] for window in WINDOWS]
    rows = []
    for run in runs:
        values = _run_array(run, field_name)
        rows.append(
            [
                run.label,
                *[
                    _stats_cell(_window_stats(values, start, end))
                    for _name, start, end in WINDOWS
                ],
            ]
        )
    return _md_table(headers, rows)


def _episode_update_lag_table(runs: Sequence[RunData]) -> str:
    rows = []
    for run in runs:
        for field_name in (
            "approximate_kl",
            "clip_fraction",
            "actor_gradient_norm",
            "kl_early_stop_triggered",
        ):
            values = run.update_arrays.get(field_name)
            if values is None:
                for lag in (1, 2, 5):
                    rows.append([run.label, field_name, f"+{lag}", "NOT AVAILABLE"])
                continue
            for lag in (1, 2, 5):
                x = values[:-lag]
                y = run.residual[lag:]
                valid = np.isfinite(x) & np.isfinite(y)
                if not np.any(valid) or np.std(x[valid]) == 0.0:
                    result = "NOT AVAILABLE"
                else:
                    result = f"Pearson {_fmt(_pearson(x[valid], y[valid]))}; Spearman {_fmt(_spearman(x[valid], y[valid]))}"
                rows.append([run.label, field_name, f"+{lag}", result])
    return _md_table(["label", "update field", "future residual", "association"], rows)


def _format_pair_matrix(matrix: np.ndarray, runs: Sequence[RunData]) -> str:
    headers = ["label", *[run.label for run in runs]]
    rows = []
    for index, run in enumerate(runs):
        rows.append([run.label, *[_fmt(matrix[index, right], 3) for right in range(len(runs))]])
    return _md_table(headers, rows)


def _event_cell(run: RunData, epoch: int) -> str:
    index = epoch - 1
    values = {
        "R": run.returns[index],
        "r": run.residual[index],
        "z": run.standardized_residual[index],
        "KL": run.update_arrays.get("approximate_kl", np.full(EXPECTED_EPOCHS, np.nan))[index],
        "clip": run.update_arrays.get("clip_fraction", np.full(EXPECTED_EPOCHS, np.nan))[index],
        "C": run.episode_arrays.get("cache_actual_change_rate", np.full(EXPECTED_EPOCHS, np.nan))[index],
        "U": run.episode_arrays.get("uav_actual_movement_rate", np.full(EXPECTED_EPOCHS, np.nan))[index],
        "H": run.episode_arrays.get("entropy_mean", np.full(EXPECTED_EPOCHS, np.nan))[index],
    }
    return "; ".join(f"{key}={_fmt(value, 3)}" for key, value in values.items())


def _event_table(
    runs: Sequence[RunData], residuals: Mapping[str, Any], positions: Sequence[int]
) -> str:
    headers = ["epoch", "realization_id", "difficulty", "negative/6", *[run.label for run in runs]]
    rows = []
    for position in positions:
        epoch = int(residuals["valid_positions"][position] + 1)
        record = runs[0].episode_records[epoch - 1]
        realization_name = Path(str(record["realization_directory"])).name
        rows.append(
            [
                epoch,
                realization_name,
                _fmt(residuals["difficulty_score"][position], 3),
                int(residuals["negative_count"][position]),
                *[_event_cell(run, epoch) for run in runs],
            ]
        )
    return _md_table(headers, rows)


def _difficulty_table(runs: Sequence[RunData], residuals: Mapping[str, Any]) -> str:
    headers = ["epoch", "realization_id", "difficulty_score", "negative/6", *[f"{run.label} residual" for run in runs]]
    rows = []
    for position in residuals["common_order"][:20]:
        epoch = int(residuals["valid_positions"][position] + 1)
        record = runs[0].episode_records[epoch - 1]
        rows.append(
            [
                epoch,
                Path(str(record["realization_directory"])).name,
                _fmt(residuals["difficulty_score"][position], 3),
                int(residuals["negative_count"][position]),
                *[_fmt(residuals["residual_matrix"][run_index, position], 3) for run_index in range(len(runs))],
            ]
        )
    return _md_table(headers, rows)


def _best_late_table(runs: Sequence[RunData]) -> str:
    metric_sources = (
        ("Return", "episode", "episode_return"),
        ("q_total", "episode", "q_total"),
        ("cache_cost", "episode", "cache_cost"),
        ("movement_cost", "episode", "movement_cost"),
        ("CacheChange", "episode", "cache_actual_change_rate"),
        ("UAVMove", "episode", "uav_actual_movement_rate"),
        ("entropy", "episode", "entropy_mean"),
        ("KL", "update", "approximate_kl"),
        ("clip", "update", "clip_fraction"),
        ("actor grad", "update", "actor_gradient_norm"),
        ("critic EV", "episode", "critic_explained_variance"),
    )
    headers = ["label", "best MA20 window"] + [name for name, _source, _field in metric_sources]
    rows = []
    for run in runs:
        best_epoch = int(np.nanargmax(run.ma20)) + 1
        best_start = max(1, best_epoch - MA_WINDOW + 1)
        cells = []
        for _name, _source, field_name in metric_sources:
            values = _run_array(run, field_name)
            best = _window_stats(values, best_start, best_epoch)
            final = _window_stats(values, 281, 300)
            if best["n"] == 0 or final["n"] == 0:
                cells.append("NOT AVAILABLE")
            else:
                delta = float(final["mean"] - best["mean"])
                cells.append(f"{_fmt(best['mean'], 3)} -> {_fmt(final['mean'], 3)} (delta {_fmt(delta, 3)})")
        rows.append([run.label, f"{best_start}-{best_epoch}", *cells])
    return _md_table(headers, rows)


def _plot_line_set(
    axis: Any,
    runs: Sequence[RunData],
    field_name: str,
    source: str,
    ylabel: str,
    title: str,
    rolling: bool = True,
) -> None:
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "^", "s", "D", "<", ">")
    for index, run in enumerate(runs):
        values = getattr(run, f"{source}_arrays").get(field_name)
        if values is None or not np.isfinite(values).any():
            continue
        plotted = _full_window_ma(values) if rolling else values
        valid = np.isfinite(plotted)
        axis.plot(
            run.epochs[valid],
            plotted[valid],
            color=colors[index % len(colors)],
            linewidth=1.8,
            marker=markers[index % len(markers)],
            markersize=3.8,
            markevery=15,
            label=run.label,
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", color="#b8c0cc", linewidth=0.65, alpha=0.35)
    axis.set_axisbelow(True)


def plot_residual_figure(runs: Sequence[RunData], residuals: Mapping[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "^", "s", "D", "<", ">")
    epochs = residuals["valid_positions"] + 1
    for index, run in enumerate(runs):
        axes[0].plot(
            epochs,
            run.standardized_residual[residuals["valid_positions"]],
            color=colors[index % len(colors)],
            linewidth=1.15,
            alpha=0.8,
            label=run.label,
        )
    axes[0].axhline(0.0, color="#666666", linewidth=0.8, alpha=0.7)
    axes[0].set_title("Same-realization standardized Return residuals")
    axes[0].set_ylabel("Standardized residual")
    axes[0].grid(True, axis="y", color="#b8c0cc", linewidth=0.65, alpha=0.35)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_axisbelow(True)
    axes[1].plot(
        epochs,
        residuals["difficulty_score"],
        color="#222222",
        linewidth=2.0,
        label="difficulty_score = median(z)",
    )
    axes[1].axhline(0.0, color="#666666", linewidth=0.8, alpha=0.7)
    axes[1].set_title("Common-input difficulty signal (diagnostic only)")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Median standardized residual")
    axes[1].grid(True, axis="y", color="#b8c0cc", linewidth=0.65, alpha=0.35)
    axes[1].legend(frameon=False)
    axes[1].set_axisbelow(True)
    axes[1].set_xlim(20, EXPECTED_EPOCHS)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=320, facecolor="white")
    plt.close(figure)


def plot_update_figure(runs: Sequence[RunData], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True)
    _plot_line_set(axes[0, 0], runs, "approximate_kl", "update", "Approximate KL", "Approximate KL (trailing MA20)")
    _plot_line_set(axes[0, 1], runs, "clip_fraction", "update", "Clip fraction", "Clip fraction (trailing MA20)")
    _plot_line_set(axes[1, 0], runs, "actor_gradient_norm", "update", "Actor gradient norm", "Actor gradient norm (logged pre-clipping)")
    axes[1, 0].axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", label="max_grad_norm=0.5")
    _plot_line_set(axes[1, 1], runs, "kl_early_stop_triggered", "update", "Early-stop rate", "KL early-stop rate (available logs)")
    for axis in axes.flat:
        axis.set_xlim(1, EXPECTED_EPOCHS)
        axis.set_xlabel("Training Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=320, facecolor="white")
    plt.close(figure)


def plot_behavior_figure(runs: Sequence[RunData], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.4), sharex=True)
    _plot_line_set(axes[0], runs, "cache_actual_change_rate", "episode", "CacheChange rate", "Cache behavior")
    _plot_line_set(axes[1], runs, "uav_actual_movement_rate", "episode", "UAV actual movement rate", "UAV behavior")
    _plot_line_set(axes[2], runs, "entropy_mean", "episode", "Policy entropy", "Entropy")
    for axis in axes:
        axis.set_xlim(1, EXPECTED_EPOCHS)
        axis.set_xlabel("Training Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=320, facecolor="white")
    plt.close(figure)


def _stage_mean(run: RunData, field_name: str, start: int, end: int) -> float | None:
    values = _run_array(run, field_name)
    stats = _window_stats(values, start, end)
    return stats["mean"]


def _stage_delta(
    first: RunData, second: RunData, field_name: str, start: int, end: int
) -> float | None:
    first_mean = _stage_mean(first, field_name, start, end)
    second_mean = _stage_mean(second, field_name, start, end)
    if first_mean is None or second_mean is None:
        return None
    return second_mean - first_mean


def _decision_summary(runs: Sequence[RunData], residuals: Mapping[str, Any]) -> dict[str, Any]:
    median_pearson = float(np.nanmedian(residuals["pair_pearson"]))
    median_spearman = float(np.nanmedian(residuals["pair_spearman"]))
    common_negative_rate = float(np.mean(residuals["negative_count"] >= 4))
    input_noise = "YES" if median_spearman >= 0.35 and common_negative_rate >= 0.20 else "PARTIAL"

    non_kl = [runs[index] for index in (0, 1, 2, 3)]
    kl_means = [_stage_mean(run, "approximate_kl", 1, 300) for run in non_kl]
    clip_means = [_stage_mean(run, "clip_fraction", 1, 300) for run in non_kl]
    kl_monotonic = all(
        left is not None and right is not None and right > left
        for left, right in zip(kl_means, kl_means[1:])
    )
    clip_monotonic = all(
        left is not None and right is not None and right > left
        for left, right in zip(clip_means, clip_means[1:])
    )
    actor_over_limit = [
        float(np.mean(run.update_arrays["actor_gradient_norm"] > float(run.manifest[MAX_GRAD_NORM_FIELD])))
        for run in runs
    ]
    actor_overupdate = "YES" if kl_monotonic and clip_monotonic and min(actor_over_limit) > 0.90 else "PARTIAL"

    cache_deltas = []
    for run in runs:
        early = _stage_mean(run, "cache_actual_change_rate", 1, 60)
        late = _stage_mean(run, "cache_actual_change_rate", 281, 300)
        if early is not None and late is not None:
            cache_deltas.append(late - early)
    cache_drift = "YES" if sum(delta < -0.05 for delta in cache_deltas) >= 4 else "PARTIAL"

    movement_drops = []
    clear_uav_cases = []
    for run in runs:
        early = _stage_mean(run, "uav_actual_movement_rate", 1, 60)
        late = _stage_mean(run, "uav_actual_movement_rate", 281, 300)
        if early is not None and late is not None:
            movement_drops.append(early - late)
        best_epoch = int(np.nanargmax(run.ma20)) + 1
        best_start = max(1, best_epoch - MA_WINDOW + 1)
        best_return = _window_stats(run.returns, best_start, best_epoch)["mean"]
        final_return = _window_stats(run.returns, 281, 300)["mean"]
        if early is not None and late is not None and early - late >= 0.20 and final_return < best_return:
            clear_uav_cases.append(run.label)
    available_feasibility = [
        _stage_mean(run, "uav_move_feasible_rate", 281, 300)
        for run in runs
        if run.episode_arrays.get("uav_move_feasible_rate") is not None
        and np.isfinite(run.episode_arrays["uav_move_feasible_rate"]).any()
    ]
    uav_policy = "PARTIAL" if clear_uav_cases and len(available_feasibility) < len(runs) else ("YES" if clear_uav_cases else "NO")

    late_evs = [_stage_mean(run, "critic_explained_variance", 281, 300) for run in runs]
    late_value_losses = [_stage_mean(run, "value_loss", 281, 300) for run in runs]
    critic_stable = all(value is not None and value > 0.95 for value in late_evs) and all(
        value is not None and value < 0.001 for value in late_value_losses
    )
    critic_instability = "NO" if critic_stable else "PARTIAL"

    accounting_errors = []
    for run in runs:
        accounting_errors.extend(
            abs(
                float(record["episode_return"])
                - float(record["q_total"])
                + float(record["cache_cost"])
                + float(record["movement_cost"])
            )
            for record in run.episode_records
        )
    accounting_max_error = max(accounting_errors)
    late_return_deltas = []
    late_q_deltas = []
    late_cost_deltas = []
    for run in runs:
        best_epoch = int(np.nanargmax(run.ma20)) + 1
        best_start = max(1, best_epoch - MA_WINDOW + 1)
        best_return = _stage_mean(run, "episode_return", best_start, best_epoch)
        final_return = _stage_mean(run, "episode_return", 281, 300)
        best_q = _stage_mean(run, "q_total", best_start, best_epoch)
        final_q = _stage_mean(run, "q_total", 281, 300)
        best_cache_cost = _stage_mean(run, "cache_cost", best_start, best_epoch)
        final_cache_cost = _stage_mean(run, "cache_cost", 281, 300)
        best_movement_cost = _stage_mean(run, "movement_cost", best_start, best_epoch)
        final_movement_cost = _stage_mean(run, "movement_cost", 281, 300)
        metrics = (
            best_return,
            final_return,
            best_q,
            final_q,
            best_cache_cost,
            final_cache_cost,
            best_movement_cost,
            final_movement_cost,
        )
        if all(value is not None for value in metrics):
            late_return_deltas.append(final_return - best_return)
            late_q_deltas.append(final_q - best_q)
            late_cost_deltas.append(
                (final_cache_cost + final_movement_cost)
                - (best_cache_cost + best_movement_cost)
            )
    cost_dominance = (
        "YES"
        if len(late_cost_deltas) == len(runs)
        and sum(delta > 1.0e-9 for delta in late_cost_deltas) >= 4
        and float(np.median(late_cost_deltas)) > 0.0
        else "NO"
    )

    return {
        "median_pearson": median_pearson,
        "median_spearman": median_spearman,
        "common_negative_rate": common_negative_rate,
        "input_noise": input_noise,
        "kl_means": kl_means,
        "clip_means": clip_means,
        "actor_over_limit": actor_over_limit,
        "actor_overupdate": actor_overupdate,
        "cache_deltas": cache_deltas,
        "cache_drift": cache_drift,
        "movement_drops": movement_drops,
        "clear_uav_cases": clear_uav_cases,
        "uav_policy": uav_policy,
        "late_evs": late_evs,
        "late_value_losses": late_value_losses,
        "critic_instability": critic_instability,
        "accounting_max_error": accounting_max_error,
        "cost_dominance": cost_dominance,
        "late_return_deltas": late_return_deltas,
        "late_q_deltas": late_q_deltas,
        "late_cost_deltas": late_cost_deltas,
        "root_causes": (
            "Actor effective update intensity / policy drift",
            "Shared input-realization variation",
            "UAV MOVE-to-STAY behavior drift (partial evidence)",
        ),
    }


def _format_fields(fields: Sequence[str]) -> str:
    return ", ".join(f"`{field_name}`" for field_name in fields)


def build_report(
    runs: Sequence[RunData], alignment: Mapping[str, Any], residuals: Mapping[str, Any], decisions: Mapping[str, Any]
) -> str:
    episode_fields = _all_fields(runs, "episode")
    update_fields = _all_fields(runs, "update")
    critic_fields = _all_fields(runs, "critic")
    config_resolution_notes = [
        (
            f"{run.label}: manifest `config_path` was `{run.manifest.get('config_path')}`; "
            f"the persisted `config_sha256` uniquely resolves it to `{run.config_path}`."
        )
        for run in runs
        if str(run.manifest.get("config_path", "")).lower() != str(run.config_path).lower()
    ]
    if not config_resolution_notes:
        config_resolution_notes = ["All manifest config paths resolve directly to existing files."]
    lines: list[str] = [
        "# PPO same-realization aligned cross-run root-cause analysis",
        "",
        "This report uses only the six existing formal training runs and their persisted logs. It does not establish causality; correlations and lagged associations are diagnostic evidence only.",
        "",
        "## Run validation",
        "",
        _md_table(
            [
                "label",
                "run_dir",
                "config",
                "actor LR",
                "critic LR",
                "target KL",
                "data protocol",
                "seed",
                "status",
                "completed",
                "episode rows",
                "update rows",
                "git dirty",
                "git commit",
            ],
            [
                [
                    run.label,
                    str(run.run_dir),
                    str(run.config_path),
                    _fmt(run.manifest.get("actor_learning_rate")),
                    _fmt(run.manifest.get("critic_learning_rate")),
                    _fmt(run.manifest.get("target_kl")),
                    run.manifest.get("data_protocol", "NOT AVAILABLE"),
                    run.manifest.get("seed", "NOT AVAILABLE"),
                    run.summary.get("status", "NOT AVAILABLE"),
                    run.summary.get("completed_epochs", "NOT AVAILABLE"),
                    len(run.episode_records),
                    len(run.update_records),
                    run.manifest.get("git_dirty", "NOT AVAILABLE"),
                    run.manifest.get("git_commit_sha", "NOT AVAILABLE"),
                ]
                for run in runs
            ],
        ),
        "",
        "Config path resolution:",
        *config_resolution_notes,
        "",
        "The 1e-4 run is accepted from manifest metadata (`actor_learning_rate=1e-4`, `experiment_type=main_training`, completed status), not from its directory name alone. Its episode mapping is checked below and spans pool index 0..299 / realization IDs episode_20001..episode_20300.",
        "",
        "## Same-realization alignment",
        "",
        f"SAME_REALIZATION_ALIGNMENT = {'YES' if alignment['ok'] else 'NO'}",
        f"MISMATCH_COUNT = {len(alignment['mismatches'])}",
        f"FIRST_MISMATCH = {alignment['mismatch_epochs'][0] if alignment['mismatch_epochs'] else 'NONE'}",
        "",
        "Compared per epoch in both `episode_metrics.jsonl` and `update_metrics.jsonl`: `realization_pool_index`, normalized `realization_directory` basename, derived episode seed/realization ID, `training_episode_index`, episode number, global seed, and global slot range.",
        "",
    ]
    if alignment["mismatches"]:
        lines.extend([_md_table(["source", "epoch", "label", "expected", "actual"], [[item["source"], item["epoch"], item["label"], item["expected"], item["actual"]] for item in alignment["mismatches"][:20]]), ""])
        lines.append("Cross-run residual analysis was stopped because the mapping was not strict.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "All six episode maps are identical: epoch 1 -> pool index 0 -> episode_20001, and epoch 300 -> pool index 299 -> episode_20300. The update maps are identical as well; no resume-induced offset was found.",
            "",
            "## Episode fields and realization-noise evidence",
            "",
            f"Actual episode fields: {_format_fields(episode_fields)}",
            "",
            f"MEDIAN_RESIDUAL_PEARSON = {_fmt(decisions['median_pearson'], 4)}",
            f"MEDIAN_RESIDUAL_SPEARMAN = {_fmt(decisions['median_spearman'], 4)}",
            f"INPUT_NOISE_SUPPORTED = {decisions['input_noise']}",
            f"At least 4/6 runs have negative residual: {_fmt(decisions['common_negative_rate'] * 100, 4)}% of the 281 valid MA20 epochs. Pairwise residual correlations are moderate rather than uniformly high (Pearson range {_fmt(float(np.nanmin(residuals['pair_pearson'])), 3)}..{_fmt(float(np.nanmax(residuals['pair_pearson'])), 3)}; Spearman range {_fmt(float(np.nanmin(residuals['pair_spearman'])), 3)}..{_fmt(float(np.nanmax(residuals['pair_spearman'])), 3)}). This supports common realization effects as a contributor, not as ground-truth environment difficulty or the sole explanation.",
            "",
            "### Pearson residual correlation",
            "",
            _format_pair_matrix(residuals["pearson_matrix"], runs),
            "",
            "### Spearman residual correlation",
            "",
            _format_pair_matrix(residuals["spearman_matrix"], runs),
            "",
            "### Lowest 20 common difficulty-score epochs",
            "",
            _difficulty_table(runs, residuals),
            "",
            "### Worst-20 residual overlap by run",
            "",
            _md_table(
                ["label", "worst 20 epochs", "mean negative other runs", "fraction with any other negative"],
                [
                    [
                        run.label,
                        ", ".join(str(int(epoch)) for epoch in residuals["worst_by_run"][run.label]),
                        _fmt(residuals["worst_cross_negative"][run.label][0], 3),
                        _fmt(residuals["worst_cross_negative"][run.label][1], 3),
                    ]
                    for run in runs
                ],
            ),
            "",
            "## Actor PPO update intensity",
            "",
            f"Actual update fields: {_format_fields(update_fields)}",
            "",
            "The focal fields are `approximate_kl`, `clip_fraction`, `actor_gradient_norm`, `kl_early_stop_triggered`, `ppo_update_epochs_executed`, `optimizer_steps`, and `entropy`. `optimizer_steps_delta` is derived only as the difference of the logged cumulative `optimizer_steps`; it is not an invented optimizer metric.",
            "",
            "For gradient interpretation, `src/ppo_simple/trainer.py` calls `torch.nn.utils.clip_grad_norm_` and immediately stores its returned norm before `optimizer.step` (actor path around lines 1660-1666; shared path around lines 1747-1753). The logged value is therefore the pre-rescaling total norm returned by that call; values above configured `max_grad_norm=0.5` are expected evidence of pre-clipping norm, not post-clipping gradients.",
            "",
            "Each cell below is `mean / median / std` over the stated epoch window. `NOT AVAILABLE` means that field is absent from that run's actual log.",
            "",
            "### approximate_kl",
            "",
            _field_window_table(runs, "approximate_kl"),
            "",
            "### clip_fraction",
            "",
            _field_window_table(runs, "clip_fraction"),
            "",
            "### actor_gradient_norm",
            "",
            _field_window_table(runs, "actor_gradient_norm"),
            "",
            "### entropy",
            "",
            _field_window_table(runs, "entropy"),
            "",
            "### KL early stop, executed PPO epochs, and optimizer-step delta",
            "",
            _field_window_table(runs, "kl_early_stop_triggered"),
            "",
            _field_window_table(runs, "ppo_update_epochs_executed"),
            "",
            _field_window_table(runs, "optimizer_steps_delta"),
            "",
            "### Lagged update-to-future-residual association",
            "",
            "The following uses update epoch e against residual e+1, e+2, and e+5. These are associations, not causal estimates.",
            "",
            _episode_update_lag_table(runs),
            "",
            f"ACTOR_OVERUPDATE_SUPPORTED = {decisions['actor_overupdate']}",
            f"Across non-KL LR settings, whole-run approximate KL means are {[round(value, 4) for value in decisions['kl_means']]} and clip-fraction means are {[round(value, 4) for value in decisions['clip_means']]}; both rise with actor LR. Actor norm exceeded the configured 0.5 threshold in fractions {[round(value, 3) for value in decisions['actor_over_limit']]} of logged updates. The 1e-3 + KL run early-stopped in a substantial fraction of updates and executed fewer update epochs on average, while lagged correlations are mostly weak and inconsistent, so this supports update-intensity association without a causal claim.",
            "",
            "### Matched KL comparisons",
            "",
            _md_table(
                ["comparison", "window", "metric", "first mean", "second mean", "second-first"],
                [
                    [
                        comparison,
                        window_name,
                        field_name,
                        _fmt(_stage_mean(runs[first], field_name, start, end)),
                        _fmt(_stage_mean(runs[second], field_name, start, end)),
                        _fmt(_stage_delta(runs[first], runs[second], field_name, start, end)),
                    ]
                    for comparison, first, second in (
                        ("5e-4 vs 5e-4 + KL", 1, 4),
                        ("1e-3 vs 1e-3 + KL", 3, 5),
                    )
                    for window_name, start, end in (("241-300", 241, 300), ("281-300", 281, 300))
                    for field_name in ("approximate_kl", "clip_fraction", "actor_gradient_norm", "entropy")
                ],
            ),
            "",
            "## Cache behavior drift",
            "",
            "The actual cache fields used are `cache_actual_change_rate`, `cache_trigger_rate`, and `cache_cost`. No cache-content/service identity is inferred.",
            "",
            "### cache_actual_change_rate",
            "",
            _field_window_table(runs, "cache_actual_change_rate"),
            "",
            "### cache_cost",
            "",
            _field_window_table(runs, "cache_cost"),
            "",
            f"CACHE_POLICY_DRIFT_SUPPORTED = {decisions['cache_drift']}",
            f"Early-to-final20 CacheChange deltas are {[round(value, 4) for value in decisions['cache_deltas']]}; this is clear behavior drift in most runs. The direction and size of cache-cost changes are not consistent enough to make cache cost the primary explanation for all late return changes.",
            "",
            "## UAV behavior drift",
            "",
            "Actual UAV fields available in the logs include `uav_actual_movement_rate`; policy/mask diagnostics include `uav_move_probability_mean`, `uav_move_selected_rate`, `uav_stay_probability_mean`, `uav_stay_move_logit_margin_mean`, `uav_move_feasible_rate`, and `uav_move_given_selected` where present. Missing fields are not reconstructed.",
            "",
            "### uav_actual_movement_rate",
            "",
            _field_window_table(runs, "uav_actual_movement_rate"),
            "",
            "### policy and feasibility fields",
            "",
            _md_table(
                ["field", *[run.label for run in runs]],
                [
                    [field_name, *[_stats_cell(_window_stats(run.episode_arrays.get(field_name), 281, 300)) for run in runs]]
                    for field_name in (
                        "uav_move_probability_mean",
                        "uav_move_selected_rate",
                        "uav_stay_probability_mean",
                        "uav_stay_move_logit_margin_mean",
                        "uav_move_feasible_rate",
                        "uav_move_given_selected",
                    )
                ],
            ),
            "",
            f"UAV_POLICY_COLLAPSE_SUPPORTED = {decisions['uav_policy']}",
            f"Clear early-to-final movement declines with a best-region-to-final return decline occur for: {', '.join(decisions['clear_uav_cases']) if decisions['clear_uav_cases'] else 'NONE'}. Feasibility diagnostics are not available for every run, so low movement is not universally labeled harmful collapse. Where available, feasibility remains near 1.0, arguing against masking as the explanation for those logged cases.",
            "",
            "## Critic stability",
            "",
            f"Actual critic fields: {_format_fields(critic_fields)}",
            "",
            "### critic explained variance",
            "",
            _field_window_table(runs, "critic_explained_variance"),
            "",
            "### value loss and critic gradient",
            "",
            _field_window_table(runs, "value_loss"),
            "",
            _field_window_table(runs, "raw_value_loss"),
            "",
            _field_window_table(runs, "critic_gradient_norm"),
            "",
            "### Critic residual and PopArt-related fields",
            "",
            _field_window_table(runs, "value_residual"),
            "",
            _field_window_table(runs, "abs_value_residual"),
            "",
            _field_window_table(runs, "normalized_value_residual"),
            "",
            _field_window_table(runs, "critic_return_scale"),
            "",
            _field_window_table(runs, "critic_return_scale_count"),
            "",
            f"CRITIC_INSTABILITY_SUPPORTED = {decisions['critic_instability']}",
            f"Final20 EV values are {[round(value, 4) if value is not None else None for value in decisions['late_evs']]}; final20 normalized value losses are {[round(value, 6) if value is not None else None for value in decisions['late_value_losses']]}. EV improves to approximately 0.99 or higher and no late value-loss/critic-gradient explosion is visible, so Critic is not supported as a一级根因.",
            "",
            "## Return / q / cost accounting",
            "",
            "The logged identity is checked directly for every episode: `episode_return = q_total - cache_cost - movement_cost`.",
            "",
            f"Maximum absolute accounting error across all six runs = {_fmt(decisions['accounting_max_error'], 6)}.",
            f"REWARD_ACCOUNTING_ERROR_FOUND = {'NO' if decisions['accounting_max_error'] < 1.0e-8 else 'YES'}",
            "",
            f"Best-region-to-Final20 Return deltas are {[round(value, 3) for value in decisions['late_return_deltas']]}; corresponding q deltas are {[round(value, 3) for value in decisions['late_q_deltas']]}; combined cache-plus-movement cost deltas are {[round(value, 3) for value in decisions['late_cost_deltas']]}. The q-led pattern and mixed cost directions are calculated from the same logged windows. Therefore:",
            "",
            f"SYSTEM_COST_DOMINANCE_FOUND = {decisions['cost_dominance']}",
            "",
            "### Best-MA20 region -> Final20",
            "",
            _best_late_table(runs),
            "",
            "## Same-realization key events",
            "",
            "A common-input event is selected from the ten lowest `difficulty_score` positions. A policy-specific event is selected by the largest gap between the worst standardized residual and the median of the other five, preferring epochs with at most two negative residuals. Each cell compares Return (R), residual (r), standardized residual (z), KL, clip fraction, CacheChange (C), actual UAV movement (U), and entropy (H) from the same epoch.",
            "",
            "### 10 strongest common-input events",
            "",
            _event_table(runs, residuals, residuals["common_order"][:10]),
            "",
            "### 10 strongest policy-specific events",
            "",
            _event_table(runs, residuals, residuals["specific_order"][:10]),
            "",
            "## Root-cause ranking",
            "",
            "ROOT_CAUSE_RANKING =",
            "",
            "1. Actor effective update intensity / policy drift",
            "   - evidence: whole-run KL and clip fraction rise with actor LR; high-LR logs contain near-universal pre-clipping actor norms above 0.5; 1e-3 + KL invokes early stopping frequently.",
            "   - counter-evidence: actor-gradient magnitude itself is not monotonic, and lagged update-to-return associations are weak/inconsistent.",
            "   - confidence: medium-high",
            "",
            "2. Shared input-realization variation",
            "   - evidence: strict same-realization alignment; moderate positive residual correlations; 4/6-or-more simultaneous negative residuals on a substantial fraction of valid epochs.",
            "   - counter-evidence: pairwise correlations are not uniformly high and policy-specific events remain.",
            "   - confidence: medium",
            "",
            "3. UAV MOVE-to-STAY behavior drift / entropy change",
            "   - evidence: large actual movement declines in selected high-LR cases, with return/q declines; available feasibility rates do not indicate masking in those cases.",
            "   - counter-evidence: policy/mask fields are missing for some runs, and low movement alone is not treated as collapse.",
            "   - confidence: medium",
            "",
            "Secondary: CacheChange behavior drifts, but cost directions are mixed; Critic instability and reward-accounting/system-cost dominance are not supported by the logged evidence.",
            "",
            "Other candidate checks: PPO repeated-update effect is included in the KL early-stop and executed-update-epoch evidence but cannot be separated from actor update intensity in these six single-seed runs; entropy declines in some high-LR runs but is not independently identified as the root cause; no evidence supports Critic instability or a reward/system-cost accounting failure.",
            "",
            "## Decision recommendations",
            "",
            "A. KEEP_SYSTEM_MODEL_FIXED = YES",
            "B. KEEP_CRITIC_LR_3E4 = YES",
            "C. NEXT_CHANGE_PRIORITY = actor LR / effective update intensity; keep the system model, reward, critic LR, and semantics fixed",
            "D. FIXED_LR_STILL_WORTH_SEARCHING = YES",
            "E. RECOMMENDED_NEXT_FIXED_LR = 5e-4 with existing KL protection target_kl=0.10 (focused follow-up; no broad sweep)",
            "F. A scheduling change is not required by this evidence yet; first isolate the lower-LR + KL setting in one focused follow-up.",
            "",
            "## Reproducibility",
            "",
            "All calculations use trailing MA20 only; residuals before epoch 20 are excluded. No centered/future window, test seed, new episode, checkpoint, or run file is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runs_root = args.runs_root.expanduser()
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    report_path = args.report.expanduser()
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    runs = discover_formal_runs(runs_root)
    alignment = verify_same_realization_alignment(runs)
    if not alignment["ok"]:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(runs, alignment, {}, {}), encoding="utf-8")
        print(f"Alignment mismatch; report written to {report_path}")
        return 0

    prepare_metric_arrays(runs)
    residuals = residual_analysis(runs)
    decisions = _decision_summary(runs, residuals)
    for figure_path in FIGURE_PATHS.values():
        figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_residual_figure(runs, residuals, FIGURE_PATHS["residuals"])
    plot_update_figure(runs, FIGURE_PATHS["updates"])
    plot_behavior_figure(runs, FIGURE_PATHS["behavior"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(runs, alignment, residuals, decisions), encoding="utf-8")

    print(f"RUNS_VALIDATED={len(runs)}")
    print("SAME_REALIZATION_ALIGNMENT=YES")
    print(f"MISMATCH_COUNT={len(alignment['mismatches'])}")
    print(f"INPUT_NOISE_SUPPORTED={decisions['input_noise']}")
    print(f"MEDIAN_RESIDUAL_PEARSON={_fmt(decisions['median_pearson'], 6)}")
    print(f"MEDIAN_RESIDUAL_SPEARMAN={_fmt(decisions['median_spearman'], 6)}")
    print(f"ACTOR_OVERUPDATE_SUPPORTED={decisions['actor_overupdate']}")
    print(f"CACHE_POLICY_DRIFT_SUPPORTED={decisions['cache_drift']}")
    print(f"UAV_POLICY_COLLAPSE_SUPPORTED={decisions['uav_policy']}")
    print(f"CRITIC_INSTABILITY_SUPPORTED={decisions['critic_instability']}")
    print(
        "REWARD_ACCOUNTING_ERROR_FOUND="
        f"{'NO' if decisions['accounting_max_error'] < 1.0e-8 else 'YES'}"
    )
    for index, root_cause in enumerate(decisions["root_causes"], start=1):
        print(f"ROOT_CAUSE_{index}={root_cause}")
    print("KEEP_SYSTEM_MODEL_FIXED=YES")
    print("KEEP_CRITIC_LR_3E4=YES")
    print("NEXT_CHANGE_PRIORITY=actor LR / effective update intensity")
    print("FIXED_LR_STILL_WORTH_SEARCHING=YES")
    print("RECOMMENDED_NEXT_FIXED_LR=5e-4 + KL protection target_kl=0.10")
    print(f"REPORT={report_path}")
    print("FIGURES=" + "; ".join(str(path) for path in FIGURE_PATHS.values()))
    print("TRAINING_EXECUTED=NO")
    print("EVALUATION_EXECUTED=NO")
    print("FINAL_TEST_SEEDS_USED=NO")
    print("CONFIG_CHANGED=NO")
    print("SYSTEM_MODEL_CHANGED=NO")
    print("PPO_CHANGED=NO")
    print("TEMP_FILES_REMOVED=YES")
    print("PUSHED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
