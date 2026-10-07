"""启动 PROPOSED 正式 300-realization 训练。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ppo_simple.trainer import main as trainer_main


def main() -> int:
    return trainer_main(
        [
            "--config",
            "configs/proposed/system_config.toml",
            "--epochs",
            "300",
            "--run-dir",
            "runs/formal_proposed",
            "--episode-pool",
            "environment_data/proposed_joint_training",
            "--episode-seed-start",
            "20001",
            "--seed",
            "27005",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
