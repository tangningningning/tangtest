"""Package entrypoint for the sole Stage3 MAPPO training implementation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


_PROJECT_ROOT_CANDIDATES = (
    Path.cwd().resolve(),
    Path(__file__).resolve().parents[2],
)
PROJECT_ROOT = next(
    (
        candidate
        for candidate in _PROJECT_ROOT_CANDIDATES
        if (candidate / "configs" / "proposed" / "system_config.toml").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not locate the adaptive_cache_mec project root. Run `ppo-train` "
        "from the project directory."
    )
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    from run_mappo_stage3_cache_value import main as stage3_main

    return stage3_main(None if argv is None else list(argv))
