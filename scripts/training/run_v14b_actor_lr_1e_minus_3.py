"""Fresh formal V14b run with Macro Actor learning rate 1e-3."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.v14b_runner import build_v14b_run_spec, main_for_run_spec

RUN_SPEC = {
    "actor_lr": 1e-3,
    "lr_label": "1e_minus_3",
    "run_name": "formal_v14b_actor_lr_1e_minus_3",
    "run_dir": "runs/formal_v14b_actor_lr_1e_minus_3",
}


def build_run_spec():
    return build_v14b_run_spec(**RUN_SPEC)


def main(argv: list[str] | None = None) -> int:
    return main_for_run_spec(build_run_spec(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
