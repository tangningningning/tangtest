"""Canonical module entry for the formal multi-node v4 PPO main experiment."""

from __future__ import annotations

from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    training_scripts = project_root / "scripts" / "training"
    if str(training_scripts) not in sys.path:
        sys.path.insert(0, str(training_scripts))
    from mainline_runner import main as run_main_experiment

    return run_main_experiment(argv)


if __name__ == "__main__":
    raise SystemExit(main())
