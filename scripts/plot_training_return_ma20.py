"""Plot formal PPO training returns with a transparent raw trace and MA20 trend.

All plotted returns are loaded from ``episode_metrics.jsonl``. The script does
not fill missing epochs or use evaluation data; the raw line is only a visual
background for the MA20 trend.

Run directories can be supplied explicitly, for example::

    python scripts/plot_training_return_ma20.py \
        --run "LR=1e-4" runs/ppo_v9_joint_cache_uav_stage1_ep50_20001_20050 \
        --run "LR=5e-4" runs/lr_sweep_proposed_joint_actor_5e4

With no ``--run`` arguments, the six formal run directories currently present
in this repository are used. The paths below are only data locations; return
values are never embedded in the script.
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
MAX_TRAINING_EPOCH = 300
MA_WINDOW = 20
RAW_RETURN_ALPHA = 0.15
MA20_MARKER_INTERVAL = 15


@dataclass(frozen=True)
class RunSpec:
    """A plotted label and the formal run directory supplying its data."""

    label: str
    run_dir: Path


DEFAULT_FORMAL_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "LR=1e-4",
        PROJECT_ROOT / "runs" / "ppo_v9_joint_cache_uav_stage1_ep50_20001_20050",
    ),
    RunSpec(
        "LR=5e-4",
        PROJECT_ROOT / "runs" / "lr_sweep_proposed_joint_actor_5e4",
    ),
    RunSpec(
        "LR=5e-4 + KL",
        PROJECT_ROOT / "runs" / "lr_sweep_v9_actor_5e4_kl_protection",
    ),
    RunSpec(
        "LR=7e-4",
        PROJECT_ROOT / "runs" / "lr_sweep_v9_actor_7e4",
    ),
    RunSpec(
        "LR=1e-3",
        PROJECT_ROOT / "runs" / "lr_sweep_v9_actor_1e3",
    ),
    RunSpec(
        "LR=1e-3 + KL",
        PROJECT_ROOT / "runs" / "lr_sweep_v9_actor_1e3_kl_protection",
    ),
)


def read_episode_returns(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate ordered epoch/return pairs from one formal run."""
    run_path = Path(run_dir)
    metrics_path = run_path / EPISODE_METRICS_FILENAME
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing formal episode metrics: {metrics_path}")

    epochs: list[int] = []
    returns: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {metrics_path}:{line_number}") from error

            if record.get("record_type", "episode") != "episode":
                continue
            if "epoch" not in record or "episode_return" not in record:
                raise ValueError(
                    f"Missing epoch/episode_return at {metrics_path}:{line_number}"
                )

            epoch_value = record["epoch"]
            if isinstance(epoch_value, bool) or not isinstance(epoch_value, (int, float)):
                raise ValueError(f"Invalid epoch at {metrics_path}:{line_number}")
            epoch = int(epoch_value)
            if epoch != epoch_value or epoch < 1 or epoch > MAX_TRAINING_EPOCH:
                raise ValueError(
                    f"Epoch outside 1-{MAX_TRAINING_EPOCH} at {metrics_path}:{line_number}"
                )

            episode_return = float(record["episode_return"])
            if not math.isfinite(episode_return):
                raise ValueError(f"Non-finite episode_return at {metrics_path}:{line_number}")

            epochs.append(epoch)
            returns.append(episode_return)

    if not epochs:
        raise ValueError(f"No episode records found in {metrics_path}")
    expected_epochs = list(range(1, len(epochs) + 1))
    if epochs != expected_epochs:
        raise ValueError(
            f"Epochs in {metrics_path} must be ordered and contiguous from 1; "
            f"received {epochs[:3]}...{epochs[-3:]}"
        )

    return np.asarray(epochs, dtype=int), np.asarray(returns, dtype=float)


def moving_average(
    values: Sequence[float] | np.ndarray, window: int = MA_WINDOW
) -> np.ndarray:
    """Return a full-window moving average, with NaN before the first window."""
    if window < 1:
        raise ValueError("window must be positive")
    numeric_values = np.asarray(values, dtype=float)
    result = np.full(numeric_values.shape, np.nan, dtype=float)
    if numeric_values.size < window:
        return result
    for index in range(window - 1, numeric_values.size):
        result[index] = float(np.mean(numeric_values[index - window + 1 : index + 1]))
    return result


def plot_training_return_ma20(
    run_specs: Iterable[RunSpec], output_path: str | Path
) -> Path:
    """Create the one-panel formal training Return comparison PNG."""
    specs = tuple(run_specs)
    if not specs:
        raise ValueError("At least one formal run is required")
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("Run labels must be unique")

    loaded_runs: list[tuple[RunSpec, np.ndarray, np.ndarray, np.ndarray]] = []
    for spec in specs:
        epochs, returns = read_episode_returns(spec.run_dir)
        ma20 = moving_average(returns, MA_WINDOW)
        if not np.isfinite(ma20).any():
            raise ValueError(f"{spec.label} has fewer than {MA_WINDOW} complete epochs")
        loaded_runs.append((spec, epochs, returns, ma20))

    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "^", "s", "D", "<", ">")

    for index, (spec, epochs, returns, ma20) in enumerate(loaded_runs):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]

        # The raw trace is deliberately excluded from the legend. It is a
        # transparent fluctuation background for the corresponding MA20 line.
        axis.plot(
            epochs,
            returns,
            color=color,
            linewidth=0.8,
            alpha=RAW_RETURN_ALPHA,
            label="_nolegend_",
        )

        valid = np.isfinite(ma20)
        axis.plot(
            epochs[valid],
            ma20[valid],
            color=color,
            linewidth=2.3,
            marker=marker,
            markersize=5.0,
            markevery=MA20_MARKER_INTERVAL,
            label=spec.label,
        )

    axis.set_xlim(1, MAX_TRAINING_EPOCH)
    axis.set_xlabel("Training Epoch")
    axis.set_ylabel("Episode Return")
    axis.set_title("PPO Training Return Comparison")
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
    plt.close(figure)
    return output


def _resolve_run_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        help="Add one formal run as LABEL RUN_DIR; repeat for multiple configurations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_return_ma20.png"),
        help="PNG output path (default: training_return_ma20.png)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.run:
        run_specs = [
            RunSpec(label=label, run_dir=_resolve_run_dir(run_dir))
            for label, run_dir in args.run
        ]
    else:
        run_specs = DEFAULT_FORMAL_RUNS

    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    created_path = plot_training_return_ma20(run_specs, output_path)
    for spec in run_specs:
        _, returns = read_episode_returns(spec.run_dir)
        print(f"{spec.label}: {returns.size} episode returns from {spec.run_dir}")
    print(f"Saved {created_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
