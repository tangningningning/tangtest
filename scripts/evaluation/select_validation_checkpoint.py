#!/usr/bin/env python3
"""Evaluate periodic checkpoints on validation and select by mean episode return."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_simple.formal_evaluation import SELECTION_METRIC, run_evaluation


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_periodic_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in run_dir.iterdir():
        match = re.fullmatch(r"epoch_(\d{3})_checkpoint\.pt", path.name)
        if match and path.is_file():
            epoch = int(match.group(1))
            if epoch <= 0 or epoch % 10:
                raise ValueError(f"invalid periodic checkpoint epoch: {path}")
            checkpoints.append((epoch, path))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError("run has no epoch_XXX_checkpoint.pt files")
    if len({epoch for epoch, _ in checkpoints}) != len(checkpoints):
        raise ValueError("duplicate periodic checkpoint epoch")
    expected_epochs = list(range(10, 301, 10))
    actual_epochs = [epoch for epoch, _ in checkpoints]
    if actual_epochs != expected_epochs:
        raise ValueError(f"formal 300-epoch periodic checkpoint sequence incomplete: {actual_epochs}")
    return checkpoints


def select_checkpoint(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("validation checkpoint rows are empty")
    if any(row["nan_inf"] or not math.isfinite(row["mean_return"]) for row in rows):
        raise ValueError("validation rows contain NaN/Inf")
    best = rows[0]
    for row in rows[1:]:
        if (row["mean_return"] > best["mean_return"] and
                not math.isclose(row["mean_return"], best["mean_return"],
                                 rel_tol=1e-12, abs_tol=1e-9)):
            best = row
    return {
        "selected_checkpoint_epoch": best["epoch"],
        "selected_checkpoint_path": best["checkpoint_path"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selection_metric": SELECTION_METRIC,
        "selection_metric_value": best["mean_return"],
        "tie_break": "earlier_epoch_for_exact_or_close_mean_return",
        "candidate_checkpoint_count": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="environment_data/proposed_joint_validation")
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    args = parser.parse_args(argv)
    run_dir, output_dir = _path(args.run_dir), _path(args.output_dir)
    checkpoints = discover_periodic_checkpoints(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("validation batch output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output_dir / "validation_checkpoint_summary.jsonl").open("w", encoding="utf-8") as stream:
        for epoch, checkpoint in checkpoints:
            summary = run_evaluation(
                checkpoint=checkpoint, split="validation",
                data_root=_path(args.data_root),
                output_dir=output_dir / f"epoch_{epoch:03d}",
                config_path=_path(args.config),
            )
            if summary["checkpoint_epoch"] != epoch:
                raise ValueError(f"checkpoint metadata epoch differs from filename: {checkpoint}")
            row = {
                "epoch": epoch, "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": summary["checkpoint_identifier"],
                "mean_return": summary["mean_episode_return"],
                "std_return": summary["std_episode_return"],
                "median_return": summary["median_episode_return"],
                "ci95_low": summary["ci95_low"], "ci95_high": summary["ci95_high"],
                "nan_inf": summary["nan_inf_detected"],
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    selection = select_checkpoint(rows)
    (output_dir / "validation_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[VALIDATION] selected epoch={selection['selected_checkpoint_epoch']} "
          f"mean_return={selection['selection_metric_value']:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
