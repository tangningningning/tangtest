"""Fresh formal V14b repeat 01."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.v14b_runner import build_v14b_run_arguments, main_for_repeat

REPEAT_ID = 1
TRAINING_SEED = 27005


def build_run_arguments(*, verbose: bool = False):
    return build_v14b_run_arguments(
        repeat_id=REPEAT_ID, seed=TRAINING_SEED, verbose=verbose
    )


def main(argv: list[str] | None = None) -> int:
    return main_for_repeat(REPEAT_ID, TRAINING_SEED, argv)


if __name__ == "__main__":
    raise SystemExit(main())
