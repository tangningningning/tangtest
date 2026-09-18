"""Plot the six complete formal PPO learning-rate runs on one MA20 chart.

Run directories are discovered from ``runs/`` using their persisted manifest,
summary, and episode metrics. No run directory or return value is selected by
plotting code through a hard-coded experiment result.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODE_METRICS_FILENAME = "episode_metrics.jsonl"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
EXPECTED_EPOCHS = 300
MA_WINDOW = 20
MA20_MARKER_INTERVAL = 15
OUTPUT_FILENAME = "training_return_lr_comparison_ma20.png"


@dataclass(frozen=True)
class TargetExperiment:
    label: str
    actor_learning_rate: float
    kl_protection: bool


@dataclass(frozen=True)
class FormalRun:
    label: str
    run_dir: Path
    config_path: Path
    epochs: np.ndarray
    returns: np.ndarray


TARGET_EXPERIMENTS: tuple[TargetExperiment, ...] = (
    TargetExperiment("1e-4", 1.0e-4, False),
    TargetExperiment("5e-4", 5.0e-4, False),
    TargetExperiment("7e-4", 7.0e-4, False),
    TargetExperiment("1e-3", 1.0e-3, False),
    TargetExperiment("5e-4 + KL", 5.0e-4, True),
    TargetExperiment("1e-3 + KL", 1.0e-3, True),
)

_EXCLUDED_RUN_NAME_TOKENS = (
    "tmp",
    "temp",
    "diagnostic",
    "probe",
    "failed",
    "failure",
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_episode_returns(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read exactly 300 ordered episode returns from one formal run."""
    metrics_path = run_dir / EPISODE_METRICS_FILENAME
    epochs: list[int] = []
    returns: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record at {metrics_path}:{line_number}")
            if record.get("record_type", "episode") != "episode":
                continue
            if "epoch" not in record or "episode_return" not in record:
                raise ValueError(f"Missing episode fields at {metrics_path}:{line_number}")

            epoch_value = record["epoch"]
            if isinstance(epoch_value, bool) or not isinstance(epoch_value, (int, float)):
                raise ValueError(f"Invalid epoch at {metrics_path}:{line_number}")
            epoch = int(epoch_value)
            if epoch != epoch_value:
                raise ValueError(f"Non-integral epoch at {metrics_path}:{line_number}")

            episode_return = float(record["episode_return"])
            if not math.isfinite(episode_return):
                raise ValueError(f"Non-finite episode_return at {metrics_path}:{line_number}")
            epochs.append(epoch)
            returns.append(episode_return)

    expected = list(range(1, EXPECTED_EPOCHS + 1))
    if epochs != expected:
        raise ValueError(
            f"{metrics_path} must contain epochs 1..{EXPECTED_EPOCHS}; "
            f"received {len(epochs)} records"
        )
    return np.asarray(epochs, dtype=np.int64), np.asarray(returns, dtype=np.float64)


def _matches_target(manifest: dict, target: TargetExperiment) -> bool:
    actor_learning_rate = manifest.get("actor_learning_rate")
    if actor_learning_rate is None:
        return False
    if not math.isclose(float(actor_learning_rate), target.actor_learning_rate, rel_tol=0.0, abs_tol=1.0e-12):
        return False

    target_kl = manifest.get("target_kl")
    if target.kl_protection:
        return target_kl is not None and math.isclose(float(target_kl), 0.10, rel_tol=0.0, abs_tol=1.0e-12)
    return target_kl is None


def _candidate_formal_run(run_dir: Path) -> tuple[dict, dict, np.ndarray, np.ndarray] | None:
    """Return complete formal-run metadata, or None for an ineligible run."""
    lowered_name = run_dir.name.lower()
    if any(token in lowered_name for token in _EXCLUDED_RUN_NAME_TOKENS):
        return None

    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    summary_path = run_dir / RUN_SUMMARY_FILENAME
    metrics_path = run_dir / EPISODE_METRICS_FILENAME
    if not all(path.is_file() for path in (manifest_path, summary_path, metrics_path)):
        return None

    try:
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        if manifest.get("experiment_type") != "main_training":
            return None
        if summary.get("status") != "completed":
            return None
        if manifest.get("target_epochs") != EXPECTED_EPOCHS:
            return None
        if summary.get("completed_epochs") != EXPECTED_EPOCHS:
            return None
        config_value = manifest.get("config_path")
        if not isinstance(config_value, str) or not config_value:
            return None
        epochs, returns = _read_episode_returns(run_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return manifest, summary, epochs, returns


def discover_formal_runs(runs_root: str | Path) -> tuple[FormalRun, ...]:
    """Discover one unique, complete formal run for every requested label."""
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {root}")

    candidates: list[tuple[Path, dict, dict, np.ndarray, np.ndarray]] = []
    for manifest_path in sorted(root.rglob(RUN_MANIFEST_FILENAME)):
        run_dir = manifest_path.parent
        candidate = _candidate_formal_run(run_dir)
        if candidate is not None:
            manifest, summary, epochs, returns = candidate
            candidates.append((run_dir, manifest, summary, epochs, returns))

    selected: list[FormalRun] = []
    for target in TARGET_EXPERIMENTS:
        matches = [
            candidate
            for candidate in candidates
            if _matches_target(candidate[1], target)
        ]
        if len(matches) != 1:
            paths = [str(candidate[0]) for candidate in matches]
            if not paths:
                paths = ["<none>"]
            raise RuntimeError(
                f"Expected exactly one complete formal run for {target.label}; "
                f"found {len(matches)}: {paths}"
            )

        run_dir, manifest, _summary, epochs, returns = matches[0]
        config_value = Path(manifest["config_path"])
        if not config_value.is_absolute():
            config_value = PROJECT_ROOT / config_value
        selected.append(
            FormalRun(
                label=target.label,
                run_dir=run_dir,
                config_path=config_value,
                epochs=epochs,
                returns=returns,
            )
        )
    return tuple(selected)


def full_window_moving_average(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return MA20, leaving positions before the first complete window empty."""
    numeric_values = np.asarray(values, dtype=np.float64)
    moving = np.full(numeric_values.shape, np.nan, dtype=np.float64)
    for index in range(MA_WINDOW - 1, numeric_values.size):
        moving[index] = float(np.mean(numeric_values[index - MA_WINDOW + 1 : index + 1]))
    return moving


def plot_training_return_lr_comparison(
    runs: Iterable[FormalRun], output_path: str | Path
) -> Path:
    """Render the single-panel MA20 comparison PNG."""
    formal_runs = tuple(runs)
    if len(formal_runs) != len(TARGET_EXPERIMENTS):
        raise ValueError(f"Expected {len(TARGET_EXPERIMENTS)} formal runs")

    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "^", "s", "D", "<", ">")
    try:
        for index, run in enumerate(formal_runs):
            ma20 = full_window_moving_average(run.returns)
            valid = np.isfinite(ma20)
            axis.plot(
                run.epochs[valid],
                ma20[valid],
                color=colors[index % len(colors)],
                linewidth=2.1,
                marker=markers[index % len(markers)],
                markersize=5.0,
                markevery=MA20_MARKER_INTERVAL,
                label=run.label,
            )

        axis.set_xlim(1, EXPECTED_EPOCHS)
        axis.set_xlabel("Training Epoch")
        axis.set_ylabel("Episode Return")
        axis.set_title("Training Return under Different Actor Learning Rates")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        axis.grid(True, axis="y", color="#b8c0cc", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.margins(y=0.08)
        axis.legend(frameon=False, loc="best", ncol=2)
        figure.tight_layout()

        output = Path(output_path)
        if output.suffix.lower() != ".png":
            raise ValueError("The output must be a PNG file")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=320, facecolor="white")
    finally:
        plt.close(figure)
    return Path(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Root to scan recursively for formal run metadata (default: runs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / OUTPUT_FILENAME,
        help=f"PNG output path (default: {OUTPUT_FILENAME})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runs_root = args.runs_root.expanduser()
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    formal_runs = discover_formal_runs(runs_root)
    for run in formal_runs:
        print(
            f"{run.label}: run_dir={run.run_dir} config={run.config_path} "
            f"epochs={run.epochs.size}"
        )
    created = plot_training_return_lr_comparison(formal_runs, output)
    print(f"Saved {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
