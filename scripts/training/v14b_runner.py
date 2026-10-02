"""Shared V14b formal runner and Actor-LR fairness preflight."""

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
REPORT_PATH = PROJECT_ROOT / "reports" / "v14b_four_actor_lr_training_preflight.json"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 300
CANONICAL_MICRO_ACTOR_LR = 3.0e-4
CANONICAL_MACRO_CRITIC_LR = 1.0e-3
EXPECTED_RUNS = (
    {
        "actor_lr": 1.0e-4,
        "lr_label": "1e_minus_4",
        "run_name": "formal_v14b_actor_lr_1e_minus_4",
        "run_dir": "runs/formal_v14b_actor_lr_1e_minus_4",
        "script": "scripts/training/run_v14b_actor_lr_1e_minus_4.py",
    },
    {
        "actor_lr": 5.0e-4,
        "lr_label": "5e_minus_4",
        "run_name": "formal_v14b_actor_lr_5e_minus_4",
        "run_dir": "runs/formal_v14b_actor_lr_5e_minus_4",
        "script": "scripts/training/run_v14b_actor_lr_5e_minus_4.py",
    },
    {
        "actor_lr": 1.0e-3,
        "lr_label": "1e_minus_3",
        "run_name": "formal_v14b_actor_lr_1e_minus_3",
        "run_dir": "runs/formal_v14b_actor_lr_1e_minus_3",
        "script": "scripts/training/run_v14b_actor_lr_1e_minus_3.py",
    },
    {
        "actor_lr": 5.0e-3,
        "lr_label": "5e_minus_3",
        "run_name": "formal_v14b_actor_lr_5e_minus_3",
        "run_dir": "runs/formal_v14b_actor_lr_5e_minus_3",
        "script": "scripts/training/run_v14b_actor_lr_5e_minus_3.py",
    },
)

for import_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def build_v14b_run_spec(
    *,
    actor_lr: float,
    lr_label: str,
    run_name: str,
    run_dir: str | Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build a fresh run with the swept rate applied only to Macro Actor."""

    resolved_run_dir = Path(run_dir).expanduser()
    if not resolved_run_dir.is_absolute():
        resolved_run_dir = PROJECT_ROOT / resolved_run_dir
    resolved_run_dir = resolved_run_dir.resolve(strict=False)
    return {
        "actor_lr": float(actor_lr),
        "lr_label": str(lr_label),
        "run_name": str(run_name),
        "run_dir": resolved_run_dir,
        "training_arguments": {
            "config_path": CONFIG_PATH,
            "target_episodes": TARGET_EPISODES,
            "run_dir": resolved_run_dir,
            "episode_pool": EPISODE_POOL,
            "episode_seed_start": EPISODE_SEED_START,
            "generate_missing_episodes": False,
            "episode_generation_device": "cpu",
            "actor_learning_rate_override": None,
            "macro_actor_learning_rate_override": float(actor_lr),
            "critic_learning_rate_override": None,
            "seed": BASE_SEED,
            "resume": False,
            "verbose": bool(verbose),
            "save_macro_transition_trace": False,
        },
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


def main_for_run_spec(spec: dict[str, Any], argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Fresh V14b Actor LR {spec['lr_label']}.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    runtime_spec = build_v14b_run_spec(
        actor_lr=spec["actor_lr"],
        lr_label=spec["lr_label"],
        run_name=spec["run_name"],
        run_dir=spec["run_dir"],
        verbose=args.verbose,
    )
    if runtime_spec["run_name"] != runtime_spec["run_dir"].name:
        raise ValueError("run_name must match the formal run directory name.")
    _assert_run_directory_is_safe(runtime_spec["run_dir"])
    from ppo_simple.trainer import run_training

    run_training(**runtime_spec["training_arguments"])
    return 0


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_hash(state_dict: dict[str, Any], prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not name.startswith(prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_ids(parameters: Any) -> set[int]:
    return {id(parameter) for parameter in parameters}


def build_preflight_report() -> dict[str, Any]:
    """Build configs/optimizers and compare initialization without rollout."""

    from common import project_config
    from ppo_simple import trainer as trainer_module
    from ppo_simple.interface import load_environment_context
    from ppo_simple.policy import DualTimescaleActorCritic
    from ppo_simple.trainer import PPOTrainer
    from scenario.realization import REALIZATION_ARTIFACT_FILENAMES

    parsed_config = project_config.parse_toml_file(CONFIG_PATH)
    ppo_config = parsed_config["algorithm"]["ppo"]
    runner_specs: list[dict[str, Any]] = []
    normalized_configs: list[dict[str, Any]] = []
    occupied_run_directories: list[str] = []
    runner_import_errors: list[str] = []
    for expected in EXPECTED_RUNS:
        script_path = PROJECT_ROOT / expected["script"]
        module_name = f"scripts.training.{script_path.stem}"
        try:
            module = importlib.import_module(module_name)
            spec = module.build_run_spec()
        except Exception as exc:
            runner_import_errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )
            continue
        spec_run_dir = Path(spec["run_dir"]).resolve(strict=False)
        if (
            float(spec["actor_lr"]) != expected["actor_lr"]
            or str(spec["lr_label"]) != expected["lr_label"]
            or str(spec["run_name"]) != expected["run_name"]
            or spec_run_dir
            != (PROJECT_ROOT / expected["run_dir"]).resolve(strict=False)
        ):
            runner_import_errors.append(f"{module_name}: RUN_SPEC differs from contract")
        args = dict(spec["training_arguments"])
        normalized_arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in args.items()
            if key != "run_dir"
        }
        normalized_arguments["macro_actor_learning_rate_override"] = "<actor_lr>"
        normalized_configs.append(
            {"canonical_config": parsed_config, "training_arguments": normalized_arguments}
        )
        if spec_run_dir.exists() and (
            not spec_run_dir.is_dir() or any(spec_run_dir.iterdir())
        ):
            occupied_run_directories.append(str(spec_run_dir))
        runner_specs.append(
            {
                "script": expected["script"],
                "actor_lr": float(spec["actor_lr"]),
                "lr_label": str(spec["lr_label"]),
                "run_name": str(spec["run_name"]),
                "run_dir": str(spec_run_dir),
                "target_episodes": int(args["target_episodes"]),
                "base_seed": int(args["seed"]),
                "episode_seed_start": int(args["episode_seed_start"]),
                "resume": bool(args["resume"]),
            }
        )

    episode_dirs = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    required_files = tuple(REALIZATION_ARTIFACT_FILENAMES.values())
    missing_realizations = [
        str(directory)
        for directory in episode_dirs
        if not directory.is_dir()
        or any(not (directory / filename).is_file() for filename in required_files)
    ]

    # This builds a static context from the fixed data; it does not step an
    # environment or execute Matching/KKT.
    context = load_environment_context(str(CONFIG_PATH), realization_dir=episode_dirs[0])
    actor_hashes: dict[str, str] = {}
    main_critic_hashes: dict[str, str] = {}
    micro_critic_hashes: dict[str, str] = {}
    long_critic_hashes: dict[str, str] = {}
    macro_actor_optimizer_lrs: dict[str, float] = {}
    micro_actor_optimizer_lrs: dict[str, float] = {}
    macro_critic_optimizer_lrs: dict[str, float] = {}
    micro_critic_optimizer_lrs: dict[str, float] = {}
    long_value_optimizer_lrs: dict[str, float] = {}
    optimizer_groups_valid: dict[str, bool] = {}
    optimizer_counters_zero: dict[str, bool] = {}
    macro_critic_shapes: list[list[int]] = []
    macro_critic_activations: list[str] = []

    for spec in runner_specs:
        run_name = spec["run_name"]
        module_name = f"scripts.training.{Path(spec['script']).stem}"
        args = importlib.import_module(module_name).build_run_spec()["training_arguments"]
        trainer_module._seed_everything(BASE_SEED)
        trainer = PPOTrainer(
            context,
            actor_learning_rate_override=args["actor_learning_rate_override"],
            macro_actor_learning_rate_override=args[
                "macro_actor_learning_rate_override"
            ],
            critic_learning_rate_override=args["critic_learning_rate_override"],
        )
        state = trainer.policy.state_dict()
        actor_prefixes = (
            "macro_actor_encoder.",
            "macro_cache_candidate_encoder.",
            "macro_cache_scorer.",
            "macro_duration_scorer.",
            "micro_actor_encoder.",
            "micro_uav_candidate_scorer.",
        )
        actor_hashes[run_name] = _state_hash(state, actor_prefixes)
        main_critic_hashes[run_name] = _state_hash(state, ("macro_critic.",))
        micro_critic_hashes[run_name] = _state_hash(state, ("micro_critic.",))
        long_critic_hashes[run_name] = _state_hash(state, ("long_value_critic.",))

        macro_actor_optimizer_lrs[run_name] = float(
            trainer.macro_optimizer.param_groups[0]["lr"]
        )
        micro_actor_optimizer_lrs[run_name] = float(
            trainer.micro_optimizer.param_groups[0]["lr"]
        )
        micro_critic_optimizer_lrs[run_name] = float(
            trainer.micro_optimizer.param_groups[1]["lr"]
        )
        macro_critic_optimizer_lrs[run_name] = float(
            trainer.macro_critic_optimizer.param_groups[0]["lr"]
        )
        long_value_optimizer_lrs[run_name] = float(
            trainer.long_value_optimizer.param_groups[0]["lr"]
        )
        if not macro_critic_shapes:
            macro_critic_shapes = [
                [int(layer.in_features), int(layer.out_features)]
                for layer in trainer.policy.macro_critic
                if hasattr(layer, "in_features") and hasattr(layer, "out_features")
            ]
            macro_critic_activations = [
                layer.__class__.__name__
                for layer in trainer.policy.macro_critic
                if not (hasattr(layer, "in_features") and hasattr(layer, "out_features"))
            ]

        macro_optimizer_ids = {
            id(parameter)
            for group in trainer.macro_optimizer.param_groups
            for parameter in group["params"]
        }
        micro_actor_group_ids = _parameter_ids(
            trainer.micro_optimizer.param_groups[0]["params"]
        )
        optimizer_groups_valid[run_name] = (
            macro_optimizer_ids == _parameter_ids(trainer.policy.macro_actor_parameters())
            and micro_actor_group_ids
            == _parameter_ids(trainer.policy.micro_actor_parameters())
            and macro_actor_optimizer_lrs[run_name] == spec["actor_lr"]
            and micro_actor_optimizer_lrs[run_name] == CANONICAL_MICRO_ACTOR_LR
            and macro_critic_optimizer_lrs[run_name] == CANONICAL_MACRO_CRITIC_LR
            and micro_critic_optimizer_lrs[run_name]
            == float(ppo_config["micro_critic_learning_rate"])
            and long_value_optimizer_lrs[run_name]
            == float(ppo_config["long_value_learning_rate"])
        )
        optimizer_counters_zero[run_name] = all(
            int(value) == 0 for value in trainer.training_counters.values()
        )

    policy_source = inspect.getsource(DualTimescaleActorCritic).lower()
    critic_fit_source = inspect.getsource(PPOTrainer._update_macro_critic_only).lower()
    macro_update_source = inspect.getsource(PPOTrainer._update_macro).lower()
    pcgrad_source = inspect.getsource(trainer_module._pcgrad_lite_current_priority).lower()
    target_scale_normalization_active = any(
        token in critic_fit_source
        for token in ("/ target_std", "/target_std", "target_std ** 2", "target_std**2")
    )
    ensemble_active = any(token in policy_source for token in ("ensemble", "macro_critics"))
    heteroscedastic_active = any(
        token in policy_source
        for token in ("heteroscedastic", "variance_head", "gaussian_nll")
    )
    last1_enabled = "last1_macro_transition_buffer" in macro_update_source
    last1_length_one = last1_enabled and "episode_index - 1" in macro_update_source
    gradient_norm_matching = (
        "current_norm_tensor / (last1_norm_raw_tensor" in pcgrad_source
        and "current_norm_tensor / mixed_norm_tensor" in pcgrad_source
    )
    current_priority_pcgrad = (
        "historical - projection_coefficient * present" in pcgrad_source
        and "current_gradient_projected" not in pcgrad_source
    )
    alpha = float(trainer_module.MACRO_CRITIC_LAST1_ALPHA)
    architecture_version = int(DualTimescaleActorCritic.architecture_version)
    checkpoint_version = int(trainer_module.CHECKPOINT_VERSION)
    target_mode = str(ppo_config.get("macro_critic_target_mode", ""))

    config_hashes = [_stable_hash(config) for config in normalized_configs]
    expected_lr_values = [float(spec["actor_lr"]) for spec in EXPECTED_RUNS]
    actual_lr_values = [float(spec["actor_lr"]) for spec in runner_specs]
    same_actor_initialization = len(set(actor_hashes.values())) == 1 and len(actor_hashes) == 4
    same_main_critic_initialization = (
        len(set(main_critic_hashes.values())) == 1 and len(main_critic_hashes) == 4
    )
    same_other_critic_initialization = (
        len(set(micro_critic_hashes.values())) == 1
        and len(set(long_critic_hashes.values())) == 1
        and len(micro_critic_hashes) == 4
        and len(long_critic_hashes) == 4
    )
    episode_sequence_identical = (
        not missing_realizations
        and all(spec["episode_seed_start"] == EPISODE_SEED_START for spec in runner_specs)
    )
    only_actor_lr_diff = (
        len(runner_specs) == len(EXPECTED_RUNS)
        and len(set(config_hashes)) == 1
        and actual_lr_values == expected_lr_values
        and all(spec["base_seed"] == BASE_SEED for spec in runner_specs)
        and all(spec["target_episodes"] == TARGET_EPISODES for spec in runner_specs)
        and all(spec["resume"] is False for spec in runner_specs)
        and all(optimizer_groups_valid.values())
    )
    checks = {
        "all_four_launchers_import": not runner_import_errors and len(runner_specs) == 4,
        "canonical_config_exists": CONFIG_PATH.is_file(),
        "training_pool_exists": EPISODE_POOL.is_dir(),
        "episode_realizations_20001_20300_complete": not missing_realizations,
        "episode_realization_sequence_identical": episode_sequence_identical,
        "run_directories_are_unique": len({spec["run_dir"] for spec in runner_specs}) == 4,
        "run_directories_are_fresh": not occupied_run_directories,
        "architecture_is_400_128_128_1_scalar_silu": (
            macro_critic_shapes == [[400, 128], [128, 128], [128, 1]]
            and macro_critic_activations == ["SiLU", "SiLU"]
        ),
        "architecture_version_is_15": architecture_version == 15,
        "checkpoint_version_is_37": checkpoint_version == 37,
        "target_mode_is_two_step_smdp_td": target_mode == "two_step_smdp_td",
        "only_macro_actor_lr_is_overridden": all(optimizer_groups_valid.values()),
        "macro_actor_lr_values_match_spec": actual_lr_values == expected_lr_values,
        "same_initial_actor_parameters": same_actor_initialization,
        "same_initial_main_critic_parameters": same_main_critic_initialization,
        "same_initial_micro_and_long_critic_parameters": same_other_critic_initialization,
        "only_actor_lr_diff": only_actor_lr_diff,
        "target_scale_normalization_disabled": not target_scale_normalization_active,
        "ensemble_disabled": not ensemble_active,
        "heteroscedastic_disabled": not heteroscedastic_active,
        "last1_length_is_one": last1_length_one,
        "gradient_norm_matching_enabled": gradient_norm_matching,
        "current_priority_pcgrad": current_priority_pcgrad,
        "alpha_is_one": alpha == 1.0,
        "target_episodes_is_300": all(
            spec["target_episodes"] == TARGET_EPISODES for spec in runner_specs
        ),
        "base_seed_is_27005": all(spec["base_seed"] == BASE_SEED for spec in runner_specs),
        "resume_is_false": all(spec["resume"] is False for spec in runner_specs),
        "optimizer_counters_untouched": all(optimizer_counters_zero.values()),
        "no_optimizer_step_called": all(optimizer_counters_zero.values()),
    }
    ready = all(checks.values())
    report = {
        "LATEST_ALGORITHM": "V14b",
        "ARCHITECTURE_VERSION": architecture_version,
        "CHECKPOINT_VERSION": checkpoint_version,
        "ACTOR_LR_VALUES": actual_lr_values,
        "SCRIPT_1": EXPECTED_RUNS[0]["script"],
        "SCRIPT_2": EXPECTED_RUNS[1]["script"],
        "SCRIPT_3": EXPECTED_RUNS[2]["script"],
        "SCRIPT_4": EXPECTED_RUNS[3]["script"],
        "RUNS": runner_specs,
        "TARGET_EPISODES": TARGET_EPISODES,
        "BASE_SEED": BASE_SEED,
        "EPISODE_SEEDS": "20001-20300",
        "EPISODE_REALIZATION_SEQUENCE_IDENTICAL": episode_sequence_identical,
        "FRESH_INITIALIZATION": True,
        "RESUME": False,
        "ACTOR_INITIAL_STATE_SHA256": next(iter(actor_hashes.values()), None),
        "ACTOR_INITIAL_STATE_SHA256_BY_RUN": actor_hashes,
        "MAIN_CRITIC_INITIAL_STATE_SHA256": next(iter(main_critic_hashes.values()), None),
        "MAIN_CRITIC_INITIAL_STATE_SHA256_BY_RUN": main_critic_hashes,
        "MICRO_CRITIC_INITIAL_STATE_SHA256_BY_RUN": micro_critic_hashes,
        "LONG_CRITIC_INITIAL_STATE_SHA256_BY_RUN": long_critic_hashes,
        "SAME_INITIAL_ACTOR_PARAMETERS": same_actor_initialization,
        "SAME_INITIAL_CRITIC_PARAMETERS": (
            same_main_critic_initialization and same_other_critic_initialization
        ),
        "ONLY_ACTOR_LR_DIFF": only_actor_lr_diff,
        "CANONICAL_CONFIG_HASH_EXCLUDING_SWEEP_FIELDS": (
            config_hashes[0] if config_hashes else None
        ),
        "MACRO_ACTOR_OPTIMIZER_LR_BY_RUN": macro_actor_optimizer_lrs,
        "MICRO_ACTOR_OPTIMIZER_LR_BY_RUN": micro_actor_optimizer_lrs,
        "MACRO_CRITIC_OPTIMIZER_LR_BY_RUN": macro_critic_optimizer_lrs,
        "MICRO_CRITIC_OPTIMIZER_LR_BY_RUN": micro_critic_optimizer_lrs,
        "LONG_VALUE_OPTIMIZER_LR_BY_RUN": long_value_optimizer_lrs,
        "LR_AFFECTED_OPTIMIZERS": ["Macro Actor optimizer (macro_optimizer)"],
        "MACRO_CRITIC_LR": float(ppo_config["macro_critic_learning_rate"]),
        "MICRO_ACTOR_LR": float(ppo_config["actor_learning_rate"]),
        "MICRO_CRITIC_LR": float(ppo_config["micro_critic_learning_rate"]),
        "LONG_VALUE_LR": float(ppo_config["long_value_learning_rate"]),
        "PPO_CLIP": float(ppo_config["clip_epsilon"]),
        "PPO_EPOCHS": int(ppo_config["update_epochs"]),
        "CRITIC_STEPS": int(ppo_config["macro_critic_steps_per_update"]),
        "MAIN_VALUE_LOSS_COEFFICIENT": float(ppo_config["value_loss_coefficient"]),
        "MACRO_CRITIC_MAX_GRAD_NORM": float(ppo_config["max_grad_norm"]),
        "TARGET_MODE": target_mode,
        "LAST1_ENABLED": last1_enabled,
        "LAST1_LENGTH": 1 if last1_length_one else 0,
        "GRADIENT_NORM_MATCHING": gradient_norm_matching,
        "CURRENT_PRIORITY_PCGRAD": current_priority_pcgrad,
        "ALPHA": alpha,
        "TARGET_SCALE_NORMALIZATION_ACTIVE": target_scale_normalization_active,
        "ENSEMBLE_ACTIVE": ensemble_active,
        "HETEROSCEDASTIC_ACTIVE": heteroscedastic_active,
        "TRAINING_EXECUTED": False,
        "ROLLOUT_EXECUTED": False,
        "OPTIMIZER_STEP_EXECUTED": False,
        "READY_FOR_FOUR_LR_TRAINING": ready,
        "CHECKS": checks,
        "BLOCKERS": [
            *runner_import_errors,
            *[f"missing/incomplete realization: {path}" for path in missing_realizations],
            *[f"run directory is non-empty: {path}" for path in occupied_run_directories],
        ],
        "MISSING_REALIZATIONS": missing_realizations,
        "OCCUPIED_RUN_DIRECTORIES": occupied_run_directories,
        "FORMAL_TRAINING_STARTED": False,
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
