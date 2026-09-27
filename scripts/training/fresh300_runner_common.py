#!/usr/bin/env python3
"""Shared implementation for independent fresh 300-epoch actor-LR runs."""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_RELATIVE = Path("configs/proposed/system_config.toml")
EPISODE_POOL_RELATIVE = Path("environment_data/proposed_joint_training")
TRAINING_START_MODE = "FRESH_INITIALIZATION"
START_EPOCH = 1
TARGET_EPOCH = 300
EPISODE_SEED_START = 20001
EPISODE_SEED_END = 20300
TARGET_CRITIC_LR = 1e-3
EXACT_Q_PARALLEL_WORKERS = 4
GPU_REQUIRED = False
CPU_TRAINING_SUPPORTED = True
RESUME = False


@dataclass(frozen=True)
class FreshRunnerSpec:
    run_name: str
    run_dir_relative: Path
    lr_label: str
    target_actor_lr: float


def _load_toml_from_text(text: str) -> dict[str, Any]:
    value = tomllib.loads(text)
    if not isinstance(value, dict):
        raise TypeError("generated TOML root is not a table")
    return value


def _replace_actor_learning_rate(text: str, target_actor_lr: float) -> str:
    table_pattern = re.compile(
        r"(?ms)(^\[algorithm\.ppo\][ \t]*\r?\n)(.*?)(?=^\[[^\]\r\n]+\][ \t]*\r?\n|\Z)"
    )
    table_match = table_pattern.search(text)
    if table_match is None:
        raise RuntimeError("algorithm.ppo table was not found")
    body = table_match.group(2)
    actor_pattern = re.compile(
        r"(?m)^(?P<prefix>[ \t]*actor_learning_rate[ \t]*=[ \t]*)"
        r"(?P<value>[^#\r\n]*?)(?P<suffix>[ \t]*(?:#[^\r\n]*)?)"
        r"(?P<newline>\r?)$"
    )
    actor_matches = list(actor_pattern.finditer(body))
    if len(actor_matches) != 1:
        raise RuntimeError(
            f"actor_learning_rate definition count is {len(actor_matches)}"
        )
    actor_literal = f"{target_actor_lr:.1e}"
    body = actor_pattern.sub(
        lambda match: (
            f"{match.group('prefix')}{actor_literal}{match.group('suffix')}"
            f"{match.group('newline')}"
        ),
        body,
        count=1,
    )
    return text[: table_match.start(2)] + body + text[table_match.end(2) :]


def _render_run_config(
    base_path: Path,
    spec: FreshRunnerSpec,
    *,
    actor_initial_state_sha256: str | None = None,
    critic_initial_state_sha256: str | None = None,
    uav_candidate_encoder_initial_state_sha256: str | None = None,
) -> str:
    text = _replace_actor_learning_rate(
        base_path.read_text(encoding="utf-8"),
        spec.target_actor_lr,
    )
    text += (
        "\n\n[training_provenance]\n"
        f'RUN_NAME = "{spec.run_name}"\n'
        f'RUN_DIRECTORY = "{spec.run_dir_relative.as_posix()}"\n'
        f'TRAINING_START_MODE = "{TRAINING_START_MODE}"\n'
        f"START_EPOCH = {START_EPOCH}\n"
        f"TARGET_EPOCH = {TARGET_EPOCH}\n"
        f'EPISODE_POOL = "{EPISODE_POOL_RELATIVE.as_posix()}"\n'
        f"EPISODE_SEED_START = {EPISODE_SEED_START}\n"
        f"EPISODE_SEED_END = {EPISODE_SEED_END}\n"
        f'LR_LABEL = "{spec.lr_label}"\n'
        f"ACTOR_LEARNING_RATE = {spec.target_actor_lr:.1e}\n"
        f"CRITIC_LEARNING_RATE = {TARGET_CRITIC_LR:.1e}\n"
        "CACHE_TRIGGER_CREDIT_SOURCE = \"FULL_INFORMATION_BEST_UPDATE_VALUE_TEACHER\"\n"
        "CACHE_CANDIDATE_CREDIT_SOURCE = \"CANDIDATE_SPECIFIC_VALUE_LISTWISE_TEACHER\"\n"
        "CACHE_VALUE_EVALUATOR = \"FULL_EXACT\"\n"
        "CACHE_DOWNSTREAM_MODE = \"FIXED_DOWNSTREAM\"\n"
        "CACHE_COUNTERFACTUAL_MODE = \"FULL_EXACT_CORRECTNESS_GATE\"\n"
        "CHECKPOINT_COMPATIBILITY = \"FRESH_INITIALIZATION_ONLY_VERSION_18\"\n"
        "UAV_STAGE1B_KL_STATISTIC = \"MEAN_OR_P90\"\n"
        f"EXACT_Q_PARALLEL_WORKERS = {EXACT_Q_PARALLEL_WORKERS}\n"
        f"GPU_REQUIRED = {str(GPU_REQUIRED).lower()}\n"
        f"CPU_TRAINING_SUPPORTED = {str(CPU_TRAINING_SUPPORTED).lower()}\n"
        f"RESUME = {str(RESUME).lower()}\n"
    )
    if actor_initial_state_sha256 is not None:
        if critic_initial_state_sha256 is None:
            raise ValueError("critic initial hash is required with actor initial hash")
        text += (
            f'ACTOR_INITIAL_STATE_SHA256 = "{actor_initial_state_sha256}"\n'
            f'CRITIC_INITIAL_STATE_SHA256 = "{critic_initial_state_sha256}"\n'
        )
        if uav_candidate_encoder_initial_state_sha256 is not None:
            text += (
                "INITIAL_UAV_CANDIDATE_ENCODER_SHA256 = "
                f'"{uav_candidate_encoder_initial_state_sha256}"\n'
            )
    _load_toml_from_text(text)
    return text


def _parameter_digest(parameters: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(f"{index}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def initial_policy_hashes(config_path: Path) -> tuple[str, str, str]:
    """Construct a fresh CPU policy and hash its actor, critic, and UAV candidate encoder."""

    source_directory = str(PROJECT_ROOT / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)

    import numpy as np
    import torch

    from common import project_config
    from ppo_simple.trainer import PPOV2Trainer, load_environment_context

    realization_directory = (
        PROJECT_ROOT
        / EPISODE_POOL_RELATIVE
        / f"episode_{EPISODE_SEED_START}"
    )
    context = load_environment_context(
        str(config_path),
        realization_dir=realization_directory,
    )
    seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = PPOV2Trainer(
        context,
        device="cpu",
        exact_q_parallel_workers=EXACT_Q_PARALLEL_WORKERS,
    )
    if trainer.optimizer_steps != 0:
        raise AssertionError("initialization validation changed optimizer steps")
    if int(getattr(trainer.policy, "architecture_version", -1)) != 2:
        raise AssertionError("formal corrected initialization requires architecture version 2")
    actor_hash = _parameter_digest(tuple(trainer.policy.actor_parameters()))
    critic_hash = _parameter_digest(tuple(trainer.policy.critic_parameters()))
    candidate_hash = _parameter_digest(
        tuple(trainer.policy.uav_candidate_encoder.parameters())
    )
    return actor_hash, critic_hash, candidate_hash


def initial_state_hashes(config_path: Path) -> tuple[str, str]:
    """Backward-compatible actor/critic hash helper used by Stage2B runners."""

    actor_hash, critic_hash, _ = initial_policy_hashes(config_path)
    return actor_hash, critic_hash


def verify_initialization(spec: FreshRunnerSpec) -> tuple[str, str]:
    """Verify one LR variant without creating a run directory or training."""

    with tempfile.TemporaryDirectory(prefix="stage2b_fresh300_config_") as temp_dir:
        config_path = Path(temp_dir) / "run_config.toml"
        config_path.write_text(
            _render_run_config(PROJECT_ROOT / BASE_CONFIG_RELATIVE, spec),
            encoding="utf-8",
            newline="",
        )
        return initial_state_hashes(config_path)


def verify_corrected_initialization(spec: FreshRunnerSpec) -> tuple[str, str, str]:
    """Verify all corrected-v2 initial parameter hashes without starting training."""

    with tempfile.TemporaryDirectory(prefix="stage2c_fresh300_config_") as temp_dir:
        config_path = Path(temp_dir) / "run_config.toml"
        config_path.write_text(
            _render_run_config(PROJECT_ROOT / BASE_CONFIG_RELATIVE, spec),
            encoding="utf-8",
            newline="",
        )
        return initial_policy_hashes(config_path)


def _config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fresh300(spec: FreshRunnerSpec) -> int:
    if len(sys.argv) > 1:
        if sys.argv[1:] == ["--verify-initialization"]:
            is_corrected_v2 = spec.run_name.startswith("stage2c_corrected_fresh300_")
            if is_corrected_v2:
                actor_hash, critic_hash, candidate_hash = verify_corrected_initialization(spec)
            else:
                actor_hash, critic_hash = verify_initialization(spec)
                candidate_hash = None
            print(f"RUN_NAME = {spec.run_name}")
            print(f"LR_LABEL = {spec.lr_label}")
            print(f"ACTOR_INITIAL_STATE_SHA256 = {actor_hash}")
            print(f"CRITIC_INITIAL_STATE_SHA256 = {critic_hash}")
            if candidate_hash is not None:
                print(f"INITIAL_UAV_CANDIDATE_ENCODER_SHA256 = {candidate_hash}")
            print("TRAINING_EXECUTED = NO")
            print("ROLLOUT_EXECUTED = NO")
            print("OPTIMIZER_STEP_CALLS = 0")
            return 0
        raise SystemExit(f"unsupported arguments: {' '.join(sys.argv[1:])}")

    base_config_path = PROJECT_ROOT / BASE_CONFIG_RELATIVE
    run_dir = PROJECT_ROOT / spec.run_dir_relative
    target_config_path = run_dir / "run_config.toml"
    if run_dir.exists() or run_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite existing fresh run directory: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        target_config_path.write_text(
            _render_run_config(base_config_path, spec),
            encoding="utf-8",
            newline="",
        )
        is_corrected_v2 = spec.run_name.startswith("stage2c_corrected_fresh300_")
        if is_corrected_v2:
            actor_hash, critic_hash, candidate_hash = initial_policy_hashes(target_config_path)
        else:
            actor_hash, critic_hash = initial_state_hashes(target_config_path)
            candidate_hash = None
        target_config_path.write_text(
            _render_run_config(
                base_config_path,
                spec,
                actor_initial_state_sha256=actor_hash,
                critic_initial_state_sha256=critic_hash,
                uav_candidate_encoder_initial_state_sha256=candidate_hash,
            ),
            encoding="utf-8",
            newline="",
        )
    except Exception:
        shutil.rmtree(run_dir)
        raise

    stage_label = (
        "STAGE2C CORRECTED"
        if spec.run_name.startswith("stage2c_corrected_fresh300_")
        else "STAGE2B"
    )
    print(
        f"[{stage_label}] TRAINING_START_MODE = {TRAINING_START_MODE}\n"
        f"[{stage_label}] RESUME = False\n"
        f"Completed Epochs: {START_EPOCH - 1}\n"
        f"Next Epoch: {START_EPOCH}\n"
        f"Target Epochs: {TARGET_EPOCH}\n"
        f"ACTOR_INITIAL_STATE_SHA256 = {actor_hash}\n"
        f"CRITIC_INITIAL_STATE_SHA256 = {critic_hash}"
        + (f"\nINITIAL_UAV_CANDIDATE_ENCODER_SHA256 = {candidate_hash}"
           if candidate_hash is not None else ""),
        flush=True,
    )

    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    source_directory = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_directory
        if not old_pythonpath
        else os.pathsep.join((source_directory, old_pythonpath))
    )
    command = [
        sys.executable,
        "-m",
        "ppo_simple.trainer",
        "--config",
        str(target_config_path),
        "--epochs",
        str(TARGET_EPOCH),
        "--run-dir",
        str(run_dir),
        "--episode-pool",
        str(EPISODE_POOL_RELATIVE),
        "--episode-seed-start",
        str(EPISODE_SEED_START),
        "--exact-q-parallel-workers",
        str(EXACT_Q_PARALLEL_WORKERS),
    ]
    if RESUME:
        raise AssertionError("fresh300 runner must not enable resume")
    trainer_process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment)
    print(f"TRAINER_PID = {trainer_process.pid}", flush=True)
    return_code = trainer_process.wait()
    if return_code != 0:
        raise SystemExit(f"trainer exited with status {return_code}")
    print(f"CONFIG_SHA256 = {_config_digest(target_config_path)}", flush=True)
    return 0
