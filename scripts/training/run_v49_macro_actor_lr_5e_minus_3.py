"""Run the V49 mainline with Macro Actor learning rate 5e-3."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.mainline_runner import main_for_run_spec

RUN_SPEC = {
    "macro_actor_learning_rate": 5e-3,
    "run_name": "formal_v49_macro_actor_lr_5e_minus_3",
}


def main() -> int:
    return main_for_run_spec(RUN_SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
