#!/usr/bin/env python3
"""Plot per-slot actual cache-state change magnitudes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ppo_simple.state_trace import read_state_trace, trace_cache_node_ids


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("plotting requires matplotlib") from exc
    return plt, MaxNLocator


def _change_magnitude(record: dict[str, Any], node_id: str) -> int:
    """Read the exact changed-bit count, with a binary-flag fallback."""

    counts = record.get("cache_update_counts")
    if isinstance(counts, dict) and node_id in counts:
        return int(counts[node_id])
    return int(record["cache_updates"].get(node_id, 0))


def _time_slots(records: list[dict[str, Any]]) -> list[int]:
    return [int(record["origin"]) for record in records]


def _style_axis(axis) -> None:
    axis.axhline(0.0, color="#334155", linewidth=0.7, alpha=0.65, zorder=1)
    axis.grid(axis="y", color="#94a3b8", linewidth=0.5, alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_cache_update_evolution(
    trace_path: str | Path,
    *,
    epoch: int | None = None,
    layout: str = "panels",
    output_dir: str | Path | None = None,
) -> Path:
    """Create Figure A from actual per-slot state differences.

    The plotted value is not cumulative.  For node ``m`` at slot ``t`` it is
    ``sum_f |C_after[m, f] - C_before[m, f]|``.
    """

    if layout not in {"panels", "overlay"}:
        raise ValueError("layout must be panels or overlay")
    path = Path(trace_path).expanduser().resolve()
    records = read_state_trace(path, epoch=epoch)
    node_ids = trace_cache_node_ids(records)
    if not node_ids:
        raise ValueError("state trace contains no caching nodes")

    plt, MaxNLocator = _matplotlib()
    output = path.parent if output_dir is None else Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "cache_update_evolution.png"
    slots = _time_slots(records)
    magnitudes = {
        node_id: np.asarray(
            [_change_magnitude(record, node_id) for record in records],
            dtype=np.float64,
        )
        for node_id in node_ids
    }
    maximum = max(float(values.max(initial=0.0)) for values in magnitudes.values())
    y_max = max(1.0, maximum * 1.15)

    if layout == "overlay":
        figure, axis = plt.subplots(figsize=(8.0, 4.6))
        for node_id, values in magnitudes.items():
            axis.plot(
                slots,
                values,
                marker="o",
                markersize=2.8,
                markevery=5,
                linewidth=1.4,
                label=node_id,
            )
        axis.set_ylim(0.0, y_max)
        axis.set_ylabel("Changed Cache Entries")
        axis.legend(frameon=False, loc="best")
        axes = [axis]
    else:
        figure, axes_array = plt.subplots(
            len(node_ids),
            1,
            sharex=True,
            figsize=(8.0, max(3.2, 2.35 * len(node_ids))),
            squeeze=False,
        )
        axes = [axes_array[index, 0] for index in range(len(node_ids))]
        for axis, node_id in zip(axes, node_ids, strict=True):
            axis.plot(
                slots,
                magnitudes[node_id],
                marker="o",
                markersize=2.8,
                markevery=5,
                linewidth=1.4,
                color="#2563eb",
            )
            axis.set_title(node_id, loc="left", pad=5.0)
            axis.set_ylabel("Changed Cache Entries")
            axis.set_ylim(0.0, y_max)

    for axis in axes:
        _style_axis(axis)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    axes[-1].set_xlabel("Time Slot")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    axes[-1].set_xlim(min(slots) - 0.5, max(slots) + 0.5)
    figure.suptitle("Cache State Update Evolution over Time", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot non-cumulative actual cache-state changes from state_trace.jsonl"
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument(
        "--layout",
        choices=("panels", "overlay"),
        default="panels",
        help="one caching-node panel by default, or overlay all node curves",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for cache_update_evolution.png; defaults to trace directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = plot_cache_update_evolution(
        args.trace,
        epoch=args.epoch,
        layout=args.layout,
        output_dir=args.output_dir,
    )
    print(f"PNG: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
