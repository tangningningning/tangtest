"""Run horizon-mask-v2 PROPOSED with Macro Actor LR 1e-4 from random initialization."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.mainline_runner import main_for_run_spec

RUN_SPEC = {
    "macro_actor_learning_rate": 1e-4,
    "run_name": "formal_cache_economic_v2_lr_1e_minus_4",
    "ppo_seed": 0,
}


def main() -> int:
    return main_for_run_spec(RUN_SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
