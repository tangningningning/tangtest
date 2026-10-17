"""Compatibility entrypoint for the sole formal MAPPO Stage3 algorithm."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
for import_root in (SOURCE_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch directly to Stage3; no legacy PPO algorithm selector remains."""

    from run_mappo_stage3_cache_value import main as stage3_main

    return stage3_main(None if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(main())