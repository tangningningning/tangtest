#!/usr/bin/env python3
"""Shared single-run formal runner for Stage2D Cache Trigger V3."""

from __future__ import annotations

import hashlib
import json
import random
import re
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
EPISODE_SEED_START = 20001
EPISODE_SEED_END = 20300
ARCHITECTURE_VERSION = 3
FRESH_INITIALIZATION = True
RESUME = False
TARGET_EPOCHS = 300
CHECKPOINT_INTERVAL_EPOCHS = 10
TARGET_CRITIC_LR = 1.0e-3
EXACT_Q_PARALLEL_WORKERS = 4

EXPECTED_RUNS = {
    "stage2d_v3_fresh300_actor_lr_1e_minus_4": 1.0e-4,
    "stage2d_v3_fresh300_actor_lr_5e_minus_4": 5.0e-4,
    "stage2d_v3_fresh300_actor_lr_1e_minus_3": 1.0e-3,
    "stage2d_v3_fresh300_actor_lr_5e_minus_3": 5.0e-3,
}


@dataclass(frozen=True)
class Stage2DV3Spec:
    run_name: str
    actor_lr: float

    @property
    def run_dir_relative(self) -> Path:
        return Path("runs") / self.run_name


def _validate_spec(spec: Stage2DV3Spec) -> None:
    expected = EXPECTED_RUNS.get(spec.run_name)
    if expected is None:
        raise ValueError(f"unknown Stage2D V3 run name: {spec.run_name}")
    if float(spec.actor_lr) != expected:
        raise ValueError(
            f"{spec.run_name} actor LR must be {expected:g}, got {spec.actor_lr:g}"
        )


def _replace_actor_learning_rate(text: str, actor_lr: float) -> str:
    table_pattern = re.compile(
        r"(?ms)(^\[algorithm\.ppo\][ \t]*\r?\n)(.*?)(?=^\[[^\]\r\n]+\][ \t]*\r?\n|\Z)"
    )
    match = table_pattern.search(text)
    if match is None:
        raise RuntimeError("[algorithm.ppo] table was not found")
    body = match.group(2)
    actor_pattern = re.compile(
        r"(?m)^(?P<prefix>[ \t]*actor_learning_rate[ \t]*=[ \t]*)"
        r"(?P<value>[^#\r\n]*?)(?P<suffix>[ \t]*(?:#[^\r\n]*)?)$"
    )
    matches = list(actor_pattern.finditer(body))
    if len(matches) != 1:
        raise RuntimeError(f"actor_learning_rate definition count is {len(matches)}")
    body = actor_pattern.sub(
        lambda item: f"{item.group('prefix')}{actor_lr:.1e}{item.group('suffix')}",
        body,
        count=1,
    )
    rendered = text[: match.start(2)] + body + text[match.end(2) :]
    return rendered


def render_run_config(
    base_path: Path,
    spec: Stage2DV3Spec,
    init_hashes: dict[str, str] | None = None,
) -> str:
    _validate_spec(spec)
    text = _replace_actor_learning_rate(
        base_path.read_text(encoding="utf-8"), spec.actor_lr
    )
    provenance = [
        "",
        "[training_provenance]",
        f'RUN_NAME = "{spec.run_name}"',
        f'RUN_DIRECTORY = "{spec.run_dir_relative.as_posix()}"',
        f"ARCHITECTURE_VERSION = {ARCHITECTURE_VERSION}",
        'TRAINING_START_MODE = "FRESH_INITIALIZATION"',
        f"FRESH_INITIALIZATION = {str(FRESH_INITIALIZATION).lower()}",
        f"RESUME = {str(RESUME).lower()}",
        f"TARGET_EPOCHS = {TARGET_EPOCHS}",
        f"CHECKPOINT_INTERVAL_EPOCHS = {CHECKPOINT_INTERVAL_EPOCHS}",
        f'EPISODE_POOL = "{EPISODE_POOL_RELATIVE.as_posix()}"',
        f"EPISODE_SEED_START = {EPISODE_SEED_START}",
        f"EPISODE_SEED_END = {EPISODE_SEED_END}",
        f"ACTOR_LR = {spec.actor_lr:.1e}",
        f"CRITIC_LR = {TARGET_CRITIC_LR:.1e}",
        "VALIDATION_ENABLED = false",
        "TEST_ENABLED = false",
        'CACHE_TRIGGER_WORKLOAD_BUILDER = "node_local_predicted_service_workload"',
        'CACHE_TRIGGER_CANDIDATE_SUMMARY = "_build_update_candidate_policy_summary"',
        'CACHE_TRIGGER_AUX_SNAPSHOT = "detached rollout snapshot"',
    ]
    if init_hashes is not None:
        required = {
            "ACTOR_INIT_SHA256",
            "CRITIC_INIT_SHA256",
            "CACHE_ENCODER_INIT_SHA256",
            "CACHE_TRIGGER_HEAD_INIT_SHA256",
            "UAV_ENCODER_INIT_SHA256",
            "UAV_CANDIDATE_ENCODER_INIT_SHA256",
        }
        if set(init_hashes) != required:
            raise ValueError("six fresh-initialization hashes are required")
        provenance.extend(f'{key} = "{init_hashes[key]}"' for key in sorted(required))
    rendered = text.rstrip() + "\n\n" + "\n".join(provenance[1:]) + "\n"
    parsed = tomllib.loads(rendered)
    ppo = parsed["algorithm"]["ppo"]
    protocol = ppo["data_protocol"]
    training = parsed["training_provenance"]
    if float(ppo["actor_learning_rate"]) != float(spec.actor_lr):
        raise AssertionError("rendered actor LR mismatch")
    if float(ppo["critic_learning_rate"]) != TARGET_CRITIC_LR:
        raise AssertionError("base critic LR differs from Stage2D requirement")
    if int(training["ARCHITECTURE_VERSION"]) != ARCHITECTURE_VERSION:
        raise AssertionError("rendered architecture version mismatch")
    if bool(training["FRESH_INITIALIZATION"]) is not True or bool(training["RESUME"]):
        raise AssertionError("Stage2D config must use fresh initialization without resume")
    if int(training["TARGET_EPOCHS"]) != TARGET_EPOCHS:
        raise AssertionError("target epoch count mismatch")
    if int(training["CHECKPOINT_INTERVAL_EPOCHS"]) != CHECKPOINT_INTERVAL_EPOCHS:
        raise AssertionError("checkpoint interval mismatch")
    if (int(protocol["train_origin_start"]), int(protocol["train_origin_end"])) != (0, 299):
        raise AssertionError("training origin range mismatch")
    if (
        protocol.get("protocol") != "full_episode_randomized"
        or int(protocol.get("episode_slots", -1)) != 300
        or int(protocol.get("rollout_steps", -1)) != 300
        or int(protocol.get("terminal_origin", -1)) != 299
    ):
        raise AssertionError("Stage2D requires complete 300-slot training realizations")
    if (
        int(training["EPISODE_SEED_START"]) != EPISODE_SEED_START
        or int(training["EPISODE_SEED_END"]) != EPISODE_SEED_END
        or training["EPISODE_POOL"] != EPISODE_POOL_RELATIVE.as_posix()
    ):
        raise AssertionError("Stage2D training pool/seed provenance mismatch")
    if bool(training["VALIDATION_ENABLED"]) or bool(training["TEST_ENABLED"]):
        raise AssertionError("Stage2D must not enable validation or test")
    if ppo.get("policy_schema") != "adaptive_joint_cache_uav_ppo":
        raise AssertionError("Stage2D config does not select the proposed joint policy")
    return rendered


def _parameter_digest(parameters: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(f"{index}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fresh_initialization_hashes(config_path: Path) -> dict[str, str]:
    """Construct only the V3 policy and critic; never load a checkpoint or roll out."""

    source_directory = str(PROJECT_ROOT / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)

    import numpy as np
    import torch

    from common import project_config
    from ppo_simple.policy import AdaptiveJointCacheUAVPPOActorCritic
    from ppo_simple.primitive_control import PrimitiveTensorSpec
    from ppo_simple.trainer import hyperparameters_from_project, load_environment_context

    realization = PROJECT_ROOT / EPISODE_POOL_RELATIVE / f"episode_{EPISODE_SEED_START}"
    if not realization.is_dir():
        raise FileNotFoundError(f"training realization missing: {realization}")
    context = load_environment_context(str(config_path), realization_dir=realization)
    if context.realization is None or Path(context.realization.directory).resolve() != realization.resolve():
        raise AssertionError("initialization preflight did not load training seed 20001")
    seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    primitive = PrimitiveTensorSpec.create(context)
    hidden_sizes = hyperparameters_from_project(context.cfg).hidden_sizes
    policy = AdaptiveJointCacheUAVPPOActorCritic(
        int(context.observation_spec.observation_dim),
        int(context.action_spec.action_dim),
        primitive.cache_feature_dim,
        primitive.uav_feature_dim,
        primitive.num_cache_heads,
        hidden_sizes,
        num_uav_heads=primitive.num_uav_heads,
    ).to(torch.device("cpu"))
    policy.eval()
    if int(policy.architecture_version) != ARCHITECTURE_VERSION:
        raise AssertionError(
            f"policy architecture version {policy.architecture_version}, expected {ARCHITECTURE_VERSION}"
        )
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError("fresh policy unexpectedly has gradients")
    return {
        "ACTOR_INIT_SHA256": _parameter_digest(tuple(policy.actor_parameters())),
        "CRITIC_INIT_SHA256": _parameter_digest(tuple(policy.critic_parameters())),
        "CACHE_ENCODER_INIT_SHA256": _parameter_digest(tuple(policy.cache_encoder.parameters())),
        "CACHE_TRIGGER_HEAD_INIT_SHA256": _parameter_digest(
            tuple(parameter for head in policy.cache_trigger_head for parameter in head.parameters())
        ),
        "UAV_ENCODER_INIT_SHA256": _parameter_digest(tuple(policy.uav_encoder.parameters())),
        "UAV_CANDIDATE_ENCODER_INIT_SHA256": _parameter_digest(
            tuple(policy.uav_candidate_encoder.parameters())
        ),
    }


def classify_run_directory(run_dir: Path) -> str:
    if not run_dir.exists():
        return "MISSING"
    if not run_dir.is_dir():
        return "UNKNOWN"
    try:
        items = list(run_dir.iterdir())
    except OSError:
        return "UNKNOWN"
    if not items:
        return "EMPTY"
    for summary_name in ("run_summary.json", "run_manifest.json"):
        summary_path = run_dir / summary_name
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return "UNKNOWN"
            try:
                completed_epochs = int(data.get("completed_epochs", 0))
            except (TypeError, ValueError):
                return "UNKNOWN"
            if data.get("status") == "completed":
                return "COMPLETE" if completed_epochs >= TARGET_EPOCHS else "PARTIAL"
            if data.get("status") in {"failed", "interrupted", "running"}:
                return "PARTIAL"
    return "PARTIAL"


def verify_training_pool() -> None:
    pool = PROJECT_ROOT / EPISODE_POOL_RELATIVE
    missing = [
        seed for seed in range(EPISODE_SEED_START, EPISODE_SEED_END + 1)
        if not (pool / f"episode_{seed}").is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "pre-generated training pool is incomplete; no data will be generated: "
            + ",".join(map(str, missing[:20]))
        )


def run_stage2d_v3(spec: Stage2DV3Spec) -> int:
    """Start exactly one manually selected formal run in the current process."""

    _validate_spec(spec)
    verify_training_pool()
    base_config = PROJECT_ROOT / BASE_CONFIG_RELATIVE
    run_dir = PROJECT_ROOT / spec.run_dir_relative
    state = classify_run_directory(run_dir)
    if state not in {"MISSING", "EMPTY"}:
        suggested = f"runs/.archive_{spec.run_name}_before_fresh_start"
        raise SystemExit(
            f"refusing to modify existing run directory ({state}): {run_dir}. "
            f"If it is an old partial/failed run, review and archive it manually as {suggested}."
        )

    with tempfile.TemporaryDirectory(prefix=".stage2d_v3_init_", dir=PROJECT_ROOT) as temp_dir:
        temp_config = Path(temp_dir) / "run_config.toml"
        temp_config.write_text(render_run_config(base_config, spec), encoding="utf-8", newline="")
        hashes = fresh_initialization_hashes(temp_config)
    target_config = render_run_config(base_config, spec, hashes)

    from ppo_simple.trainer import CHECKPOINT_INTERVAL_EPOCHS, main as trainer_main

    if int(CHECKPOINT_INTERVAL_EPOCHS) != 10:
        raise AssertionError("formal trainer checkpoint interval is not 10")
    if run_dir.exists() and classify_run_directory(run_dir) != "EMPTY":
        raise SystemExit(f"run directory changed during preflight; refusing to start: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    target_config_path = run_dir / "run_config.toml"
    if target_config_path.exists():
        raise SystemExit(f"refusing to overwrite existing run config: {target_config_path}")
    target_config_path.write_text(target_config, encoding="utf-8", newline="")

    argv = [
        "--config", str(target_config_path),
        "--epochs", str(TARGET_EPOCHS),
        "--run-dir", str(run_dir),
        "--episode-pool", EPISODE_POOL_RELATIVE.as_posix(),
        "--episode-seed-start", str(EPISODE_SEED_START),
        "--exact-q-parallel-workers", str(EXACT_Q_PARALLEL_WORKERS),
    ]
    return int(trainer_main(argv))
