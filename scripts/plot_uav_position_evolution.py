#!/usr/bin/env python3
"""Plot actual UAV x/y position evolution from a state trace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppo_simple.state_trace import read_state_trace, trace_uav_ids


POSITION_CHANGE_TOLERANCE_M = 1.0e-6


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("plotting requires matplotlib") from exc
    return plt, MaxNLocator


def _position_series(
    records: list[dict],
    uav_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[tuple[float, float]] = []
    for record in records:
        raw_position = record["uav_positions"].get(uav_id)
        if raw_position is None:
            raise ValueError(f"state trace is missing {uav_id} at origin {record['origin']}")
        values.append((float(raw_position["x"]), float(raw_position["y"])))
    positions = np.asarray(values, dtype=np.float64)
    distances = np.zeros(len(positions), dtype=np.float64)
    if len(positions) > 1:
        distances[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return positions[:, 0], positions[:, 1], distances


def _style_axis(axis) -> None:
    axis.grid(axis="y", color="#94a3b8", linewidth=0.5, alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_uav_position_evolution(
    trace_path: str | Path,
    *,
    epoch: int | None = None,
    output_dir: str | Path | None = None,
    trajectory_2d: bool = False,
    show_move_markers: bool = True,
) -> tuple[Path, Path | None]:
    """Create Figure B and optionally a supplementary 2-D trajectory PNG."""

    path = Path(trace_path).expanduser().resolve()
    records = read_state_trace(path, epoch=epoch)
    uav_ids = trace_uav_ids(records)
    if not uav_ids:
        raise ValueError("state trace contains no UAV positions")

    plt, MaxNLocator = _matplotlib()
    output = path.parent if output_dir is None else Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "uav_position_evolution.png"
    slots = np.asarray([int(record["origin"]) for record in records], dtype=np.int64)

    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        uav_id: _position_series(records, uav_id) for uav_id in uav_ids
    }
    figure, axes_array = plt.subplots(
        len(uav_ids),
        1,
        sharex=True,
        figsize=(8.0, max(3.2, 2.8 * len(uav_ids))),
        squeeze=False,
    )
    axes = [axes_array[index, 0] for index in range(len(uav_ids))]
    colors = {"x": "#2563eb", "y": "#dc2626"}
    for axis, uav_id in zip(axes, uav_ids, strict=True):
        x_values, y_values, distances = series[uav_id]
        axis.plot(
            slots,
            x_values,
            color=colors["x"],
            linewidth=1.45,
            marker="o",
            markersize=2.6,
            markevery=5,
            label="x-position",
        )
        axis.plot(
            slots,
            y_values,
            color=colors["y"],
            linewidth=1.45,
            marker="o",
            markersize=2.6,
            markevery=5,
            label="y-position",
        )
        if show_move_markers and len(slots) > 1:
            moved = distances > POSITION_CHANGE_TOLERANCE_M
            if np.any(moved):
                axis.scatter(
                    slots[moved],
                    x_values[moved],
                    color=colors["x"],
                    s=15,
                    zorder=4,
                )
                axis.scatter(
                    slots[moved],
                    y_values[moved],
                    color=colors["y"],
                    s=15,
                    zorder=4,
                )
        axis.set_title(uav_id, loc="left", pad=5.0)
        axis.set_ylabel("Position")
        axis.legend(loc="best", frameon=False, ncol=2)
        _style_axis(axis)

    axes[-1].set_xlabel("Time Slot")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    axes[-1].set_xlim(float(slots.min()) - 0.5, float(slots.max()) + 0.5)
    figure.suptitle("Adaptive UAV Position Evolution over Time", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(figure)

    trajectory_path: Path | None = None
    if trajectory_2d:
        trajectory_path = _plot_trajectory_2d(records, uav_ids, series, output, plt)
    return output_path, trajectory_path


def _plot_trajectory_2d(
    records: list[dict],
    uav_ids: list[str],
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
    plt,
) -> Path:
    output_path = output / "uav_trajectory_2d.png"
    figure, axis = plt.subplots(figsize=(7.0, 5.4))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, uav_id in enumerate(uav_ids):
        x_values, y_values, _ = series[uav_id]
        color = color_cycle[index % len(color_cycle)] if color_cycle else None
        (line,) = axis.plot(
            x_values,
            y_values,
            linewidth=1.45,
            marker="o",
            markersize=2.6,
            markevery=5,
            color=color,
            label=uav_id,
        )
        axis.scatter(
            [x_values[0]],
            [y_values[0]],
            color=line.get_color(),
            marker="o",
            s=30,
            zorder=4,
        )
        axis.scatter(
            [x_values[-1]],
            [y_values[-1]],
            color=line.get_color(),
            marker="s",
            s=34,
            zorder=4,
        )
        axis.annotate(
            "start",
            (x_values[0], y_values[0]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
        axis.annotate(
            "end",
            (x_values[-1], y_values[-1]),
            xytext=(4, -11),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_title("UAV Trajectory")
    axis.set_xlabel("x-position")
    axis.set_ylabel("y-position")
    axis.grid(color="#94a3b8", linewidth=0.5, alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if len(uav_ids) > 1:
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot actual UAV x(t)/y(t) positions from state_trace.jsonl"
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument(
        "--trajectory-2d",
        action="store_true",
        help="also write uav_trajectory_2d.png",
    )
    marker_group = parser.add_mutually_exclusive_group()
    marker_group.add_argument(
        "--show-move-markers",
        dest="show_move_markers",
        action="store_true",
        default=True,
        help="mark slots with a non-zero actual position displacement (default)",
    )
    marker_group.add_argument(
        "--no-move-markers",
        dest="show_move_markers",
        action="store_false",
        help="disable actual-displacement markers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for PNG output; defaults to trace directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path, trajectory_path = plot_uav_position_evolution(
        args.trace,
        epoch=args.epoch,
        output_dir=args.output_dir,
        trajectory_2d=args.trajectory_2d,
        show_move_markers=args.show_move_markers,
    )
    print(f"PNG: {output_path}")
    if trajectory_path is not None:
        print(f"2D PNG: {trajectory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
