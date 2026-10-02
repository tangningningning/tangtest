"""Shared formal V14b runner configuration and non-rollout preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
EPISODE_POOL = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
REPORT_PATH = PROJECT_ROOT / "reports" / "v14b_training_preflight.json"
TARGET_EPISODES = 300
EPISODE_SEED_START = 20001
REPEAT_SEEDS = {1: 27005, 2: 27006, 3: 27007, 4: 27008}
FORMAL_TRAINING_STARTED = False

for import_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def run_name(repeat_id: int, seed: int) -> str:
    if repeat_id not in REPEAT_SEEDS:
        raise ValueError(f"repeat_id must be in {sorted(REPEAT_SEEDS)}")
    return f"formal_v14b_repeat{repeat_id:02d}_seed{int(seed)}"


def build_v14b_run_arguments(
    *, repeat_id: int, seed: int, verbose: bool = False
) -> dict[str, Any]:
    """Build one fresh run from the shared, unchanged formal TOML config."""

    if REPEAT_SEEDS.get(int(repeat_id)) != int(seed):
        raise ValueError("V14b repeat id and formal training seed do not match.")
    return {
        "config_path": CONFIG_PATH,
        "target_episodes": TARGET_EPISODES,
        "run_dir": PROJECT_ROOT / "runs" / run_name(int(repeat_id), int(seed)),
        "episode_pool": EPISODE_POOL,
        "episode_seed_start": EPISODE_SEED_START,
        "generate_missing_episodes": False,
        "episode_generation_device": "cpu",
        "actor_learning_rate_override": None,
        "macro_actor_learning_rate_override": None,
        "critic_learning_rate_override": None,
        "seed": int(seed),
        "resume": False,
        "verbose": bool(verbose),
        "save_macro_transition_trace": False,
    }


def _assert_run_directory_is_safe(run_dir: Path) -> None:
    resolved = Path(run_dir).resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if resolved.parent != runs_root:
        raise ValueError(f"formal run directory is outside runs/: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"formal run path exists and is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"formal run directory is not empty; refusing to overwrite: {resolved}"
            )


def main_for_repeat(repeat_id: int, seed: int, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Fresh V14b repeat {repeat_id:02d}.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    run_arguments = build_v14b_run_arguments(
        repeat_id=repeat_id, seed=seed, verbose=args.verbose
    )
    _assert_run_directory_is_safe(run_arguments["run_dir"])

    from ppo_simple.trainer import run_training

    run_training(**run_arguments)
    return 0


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_preflight_report() -> dict[str, Any]:
    """Validate runner/config contracts without constructing an environment."""

    from common import project_config
    from ppo_simple import trainer as trainer_module
    from ppo_simple.policy import DualTimescaleActorCritic
    from ppo_simple.trainer import PPOTrainer
    from scenario.realization import REALIZATION_ARTIFACT_FILENAMES

    parsed_config = project_config.parse_toml_file(CONFIG_PATH)
    ppo_config = parsed_config["algorithm"]["ppo"]
    runner_specs: list[dict[str, Any]] = []
    config_hashes: list[str] = []
    occupied_run_directories: list[str] = []
    for repeat_id, expected_seed in REPEAT_SEEDS.items():
        module = importlib.import_module(
            f"scripts.training.run_v14b_repeat_{repeat_id:02d}"
        )
        if int(module.REPEAT_ID) != repeat_id or int(module.TRAINING_SEED) != expected_seed:
            raise ValueError(f"runner {repeat_id:02d} has an unexpected repeat/seed.")
        arguments = module.build_run_arguments()
        if arguments["config_path"] != CONFIG_PATH:
            raise ValueError(f"runner {repeat_id:02d} does not use the canonical config.")
        run_dir = Path(arguments["run_dir"]).resolve(strict=False)
        run_name_value = run_name(repeat_id, expected_seed)
        shared_arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in arguments.items()
            if key not in {"seed", "run_dir"}
        }
        config_hashes.append(
            _stable_hash({"canonical_config": parsed_config, "run_arguments": shared_arguments})
        )
        if run_dir.exists() and (
            not run_dir.is_dir() or any(run_dir.iterdir())
        ):
            occupied_run_directories.append(str(run_dir))
        runner_specs.append(
            {
                "repeat_id": repeat_id,
                "script": f"scripts/training/run_v14b_repeat_{repeat_id:02d}.py",
                "seed": expected_seed,
                "run_name": run_name_value,
                "run_dir": str(run_dir),
                "target_episodes": int(arguments["target_episodes"]),
                "resume": bool(arguments["resume"]),
            }
        )

    episode_dirs = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    required_realization_files = tuple(REALIZATION_ARTIFACT_FILENAMES.values())
    missing_realizations = [
        str(directory)
        for directory in episode_dirs
        if not directory.is_dir()
        or any(not (directory / filename).is_file() for filename in required_realization_files)
    ]

    # Construct only a policy module to verify the fixed 400-128-128-1 scalar head.
    # No environment, simulator, rollout, Matching, or KKT object is created here.
    policy = DualTimescaleActorCritic(
        observation_dim=400,
        macro_cache_feature_dim=1,
        micro_uav_feature_dim=1,
        duration_candidates=(1,),
        prediction_horizon=1,
        hidden_sizes=tuple(int(size) for size in ppo_config["hidden_sizes"]),
    )
    macro_linear_shapes = [
        [int(layer.in_features), int(layer.out_features)]
        for layer in policy.macro_critic
        if layer.__class__.__name__ == "Linear"
    ]
    macro_activations = [
        layer.__class__.__name__
        for layer in policy.macro_critic
        if layer.__class__.__name__ != "Linear"
    ]
    policy_source = inspect.getsource(DualTimescaleActorCritic).lower()
    critic_fit_source = inspect.getsource(PPOTrainer._update_macro_critic_only).lower()
    macro_update_source = inspect.getsource(PPOTrainer._update_macro).lower()
    pcgrad_source = inspect.getsource(trainer_module._pcgrad_lite_current_priority).lower()
    target_scale_normalization_active = any(
        token in critic_fit_source
        for token in ("/ target_std", "/target_std", "target_std ** 2", "target_std**2")
    )
    ensemble_active = any(
        token in policy_source for token in ("ensemble", "macro_critics")
    )
    heteroscedastic_active = any(
        token in policy_source for token in ("heteroscedastic", "variance_head", "gaussian_nll")
    )
    last1_active = "last1_macro_transition_buffer" in macro_update_source
    gradient_norm_matching = (
        "current_norm_tensor / (last1_norm_raw_tensor" in pcgrad_source
        and "current_norm_tensor / mixed_norm_tensor" in pcgrad_source
    )
    pcgrad_current_priority = (
        "historical - projection_coefficient * present" in pcgrad_source
        and "current_gradient_projected" not in pcgrad_source
    )
    last1_length_one = last1_active and "episode_index - 1" in macro_update_source
    alpha = float(trainer_module.MACRO_CRITIC_LAST1_ALPHA)
    target_mode = str(ppo_config.get("macro_critic_target_mode", ""))
    target_episodes = all(item["target_episodes"] == TARGET_EPISODES for item in runner_specs)
    resume_disabled = all(item["resume"] is False for item in runner_specs)
    output_dirs_unique = len({item["run_dir"] for item in runner_specs}) == 4
    config_identical = len(set(config_hashes)) == 1
    learning_rate_sweep = any("lr_1e" in item["script"] or "lr_3e" in item["script"] for item in runner_specs)

    checks = {
        "canonical_config_exists": CONFIG_PATH.is_file(),
        "training_pool_exists": EPISODE_POOL.is_dir(),
        "all_300_training_realizations_complete": not missing_realizations,
        "formal_run_directories_are_fresh": not occupied_run_directories,
        "architecture_is_400_128_128_1_scalar_silu": (
            macro_linear_shapes == [[400, 128], [128, 128], [128, 1]]
            and macro_activations == ["SiLU", "SiLU"]
        ),
        "architecture_version_is_15": int(DualTimescaleActorCritic.architecture_version) == 15,
        "checkpoint_version_is_37": int(trainer_module.CHECKPOINT_VERSION) == 37,
        "target_mode_is_two_step_smdp_td": target_mode == "two_step_smdp_td",
        "target_scale_normalization_disabled": not target_scale_normalization_active,
        "ensemble_disabled": not ensemble_active,
        "heteroscedastic_disabled": not heteroscedastic_active,
        "last1_enabled": last1_active and last1_length_one,
        "last1_length_is_one": last1_length_one,
        "last1_norm_matching_enabled": gradient_norm_matching,
        "pcgrad_current_priority": pcgrad_current_priority,
        "alpha_is_one": alpha == 1.0,
        "macro_critic_steps_is_50": int(ppo_config["macro_critic_steps_per_update"]) == 50,
        "macro_critic_value_loss_coefficient_is_0_01": float(ppo_config["value_loss_coefficient"]) == 0.01,
        "max_grad_norm_is_0_5": float(ppo_config["max_grad_norm"]) == 0.5,
        "target_episodes_is_300": target_episodes,
        "resume_is_false": resume_disabled,
        "run_directories_are_unique": output_dirs_unique,
        "config_identical_except_seed_and_run_id": config_identical,
        "learning_rate_sweep_disabled": not learning_rate_sweep,
    }
    ready = all(checks.values())
    report = {
        "LATEST_ALGORITHM": "V14b",
        "ARCHITECTURE_VERSION": int(DualTimescaleActorCritic.architecture_version),
        "CHECKPOINT_VERSION": int(trainer_module.CHECKPOINT_VERSION),
        "MAIN_TRAINING_SCRIPTS": [item["script"] for item in runner_specs],
        "REPEAT_SEEDS": [item["seed"] for item in runner_specs],
        "RUNS": runner_specs,
        "TARGET_EPISODES": TARGET_EPISODES,
        "LEARNING_RATE_SWEEP": learning_rate_sweep,
        "CONFIG_IDENTICAL_EXCEPT_SEED_AND_RUN_ID": config_identical,
        "CONFIG_HASH_EXCLUDING_SEED_AND_RUN_ID": config_hashes[0] if config_hashes else None,
        "CANONICAL_CONFIG_PATH": str(CONFIG_PATH),
        "CANONICAL_CONFIG_RAW_SHA256": project_config.raw_config_sha256(CONFIG_PATH),
        "TARGET_MODE": target_mode,
        "TARGET_SCALE_NORMALIZATION_ACTIVE": target_scale_normalization_active,
        "ENSEMBLE_ACTIVE": ensemble_active,
        "HETEROSCEDASTIC_ACTIVE": heteroscedastic_active,
        "LAST1_ENABLED": last1_active,
        "LAST1_LENGTH": 1 if last1_length_one else 0,
        "GRADIENT_NORM_MATCHING": gradient_norm_matching,
        "PCGRAD_CURRENT_PRIORITY": pcgrad_current_priority,
        "ALPHA": alpha,
        "MACRO_CRITIC_LEARNING_RATE": float(ppo_config["macro_critic_learning_rate"]),
        "MACRO_CRITIC_STEPS_PER_UPDATE": int(ppo_config["macro_critic_steps_per_update"]),
        "MACRO_CRITIC_VALUE_LOSS_COEFFICIENT": float(ppo_config["value_loss_coefficient"]),
        "MACRO_CRITIC_MAX_GRAD_NORM": float(ppo_config["max_grad_norm"]),
        "LEARNING_RATES": {
            "actor": float(ppo_config["actor_learning_rate"]),
            "macro_critic": float(ppo_config["macro_critic_learning_rate"]),
            "micro_actor": float(ppo_config["actor_learning_rate"]),
            "micro_critic": float(ppo_config["micro_critic_learning_rate"]),
            "long_value": float(ppo_config["long_value_learning_rate"]),
        },
        "PPO": {
            "clip_epsilon": float(ppo_config["clip_epsilon"]),
            "update_epochs": int(ppo_config["update_epochs"]),
            "target_kl": float(ppo_config["target_kl"]),
            "num_minibatches": int(ppo_config["num_minibatches"]),
            "minibatch_size": int(ppo_config["minibatch_size"]),
        },
        "SEED_PROTOCOL": {
            "model_and_training_random_seed": "REPEAT_SEEDS",
            "episode_realization_seed_start": EPISODE_SEED_START,
            "episode_realization_count": TARGET_EPISODES,
            "episode_realization_pool_shared_across_repeats": True,
            "generate_missing_episodes": False,
        },
        "FORMAL_TRAINING_STARTED": FORMAL_TRAINING_STARTED,
        "READY_FOR_FORMAL_TRAINING": ready,
        "CHECKS": checks,
        "BLOCKERS": [
            f"missing/incomplete training realization: {path}"
            for path in missing_realizations
        ]
        + [f"formal run directory is non-empty: {path}" for path in occupied_run_directories],
        "MISSING_TRAINING_REALIZATIONS": missing_realizations,
        "OCCUPIED_RUN_DIRECTORIES": occupied_run_directories,
        "FORMAL_ARTIFACTS_PER_RUN": [
            "run_config.toml",
            "run_manifest.json",
            "run_summary.json",
            "episode_metrics.jsonl",
            "update_metrics.jsonl",
            "critic_diagnostics.jsonl",
            "slot_metrics.jsonl",
            "checkpoints/latest_checkpoint.pt",
            "checkpoints/best_train_return_checkpoint.pt",
        ],
    }
    return report


def write_preflight_report() -> dict[str, Any]:
    report = build_preflight_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
