"""以共享 PROPOSED runner 启动 Macro Actor 1e-4→2e-5 cosine LR run。"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
RUN_NAME = "formal_proposed_lr_1e_minus_4"
RUN_DIRECTORY = f"runs/{RUN_NAME}"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ppo_simple.trainer import main as trainer_main


def _config_text_with_lr_sensitivity() -> str:
    """仅在临时配置副本中覆盖 Macro Actor cosine schedule 两个端点。"""

    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    replacements = (
        ("macro_actor_learning_rate", "5.0e-4", "1.0e-4"),
        ("macro_actor_final_learning_rate", "1.0e-4", "2.0e-5"),
    )
    for key, original, replacement in replacements:
        pattern = re.compile(
            rf"(?m)^([ \t]*{re.escape(key)}[ \t]*=[ \t]*){re.escape(original)}([ \t]*)$"
        )
        config_text, count = pattern.subn(rf"\g<1>{replacement}\g<2>", config_text)
        if count != 1:
            raise ValueError(
                f"正式配置中的 {key} 应恰好为 {original} 且只出现一次。"
            )
    return config_text


def _trainer_arguments(config_path: str) -> list[str]:
    return [
        "--config",
        config_path,
        "--epochs",
        "300",
        "--run-dir",
        RUN_DIRECTORY,
        "--episode-pool",
        "environment_data/proposed_joint_training",
        "--episode-seed-start",
        "20001",
        "--seed",
        "27005",
    ]


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix=".proposed_lr_1e_minus_4_", dir=CONFIG_PATH.parent
    ) as temporary_directory:
        temporary_config = Path(temporary_directory) / "system_config.toml"
        temporary_config.write_text(
            _config_text_with_lr_sensitivity(), encoding="utf-8"
        )
        config_argument = temporary_config.relative_to(PROJECT_ROOT).as_posix()
        return trainer_main(_trainer_arguments(config_argument))


if __name__ == "__main__":
    raise SystemExit(main())
