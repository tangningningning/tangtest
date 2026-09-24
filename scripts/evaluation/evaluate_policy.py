#!/usr/bin/env python3
"""Evaluate one frozen corrected PPO checkpoint on a declared full-episode split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_simple.formal_evaluation import run_evaluation


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/proposed/system_config.toml")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--selection-metadata")
    parser.add_argument("--smoke-one", action="store_true",
                        help="evaluate only the first realization for protocol checks")
    args = parser.parse_args(argv)
    if not args.deterministic:
        parser.error("formal evaluation requires deterministic actions")
    summary = run_evaluation(
        checkpoint=_path(args.checkpoint), split=args.split,
        data_root=_path(args.data_root), output_dir=_path(args.output_dir),
        config_path=_path(args.config), smoke_one=args.smoke_one,
        selection_metadata=(_path(args.selection_metadata) if args.selection_metadata else None),
    )
    print(f"[EVAL] split={summary['split']} realizations={summary['num_realizations']} "
          f"mean_episode_return={summary['mean_episode_return']:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
