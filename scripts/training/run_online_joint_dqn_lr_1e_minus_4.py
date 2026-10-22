"""使用固定学习率 1e-4 启动共享的在线联合 DQN 训练入口。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from online_joint_dqn.run_training import main as shared_training_main

LEARNING_RATE = 1e-4
RUN_NAME_PREFIX = "online_joint_dqn_lr_1e-4"


def main() -> int:
    return shared_training_main(
        learning_rate=LEARNING_RATE,
        run_name_prefix=RUN_NAME_PREFIX,
    )


if __name__ == "__main__":
    raise SystemExit(main())
