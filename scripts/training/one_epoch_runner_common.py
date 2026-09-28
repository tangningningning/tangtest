from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import project_config
from ppo_simple.trainer import PPOV2Trainer, load_environment_context, run_training


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "proposed" / "system_config.toml"
EPISODE_POOL = ROOT / "environment_data" / "proposed_joint_training"
VERIFY_REALIZATION = EPISODE_POOL / "episode_20001"
CRITIC_LR = 1.0e-3


def _hash_tensors(values: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values, key=lambda item: item[0]):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_hashes(trainer: PPOV2Trainer) -> dict[str, str | None]:
    policy = trainer.policy
    named = dict(policy.named_parameters())
    actor_ids = {id(parameter) for parameter in policy.actor_parameters()}
    actor_values = [
        (name, parameter)
        for name, parameter in named.items()
        if id(parameter) in actor_ids
    ]
    if len(actor_values) != len(actor_ids):
        raise RuntimeError("actor hash does not cover every actor parameter")

    critic_modules = [
        getattr(policy, name)
        for name in ("critic_encoder", "critic_head")
        if getattr(policy, name, None) is not None
    ]
    critic_ids = {
        id(parameter)
        for module in critic_modules
        for parameter in module.parameters()
    }
    critic_values = [
        (name, parameter)
        for name, parameter in named.items()
        if id(parameter) in critic_ids
    ]
    if not critic_values or len(critic_values) != len(critic_ids):
        raise RuntimeError("critic hash does not cover every critic parameter")
    if actor_ids & critic_ids:
        raise RuntimeError("actor and critic parameter groups overlap")

    surrogate = trainer.cache_surrogate_ensemble
    if surrogate is None:
        raise RuntimeError("proposed policy did not initialize its Cache surrogate")
    surrogate_values = list(surrogate.state_dict().items())

    candidate_encoder = getattr(policy, "uav_candidate_encoder", None)
    candidate_values = (
        None if candidate_encoder is None else list(candidate_encoder.state_dict().items())
    )
    return {
        "actor_sha256": _hash_tensors(actor_values),
        "critic_sha256": _hash_tensors(critic_values),
        "cache_surrogate_sha256": _hash_tensors(surrogate_values),
        "uav_candidate_encoder_sha256": (
            None if candidate_values is None else _hash_tensors(candidate_values)
        ),
    }


def _initialize_trainer(actor_lr: float) -> PPOV2Trainer:
    context = load_environment_context(
        str(CONFIG_PATH), realization_dir=str(VERIFY_REALIZATION)
    )
    seed = int(project_config.get_config_value(context.cfg, "experiment.seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return PPOV2Trainer(
        context,
        actor_learning_rate_override=float(actor_lr),
        critic_learning_rate_override=CRITIC_LR,
        exact_q_parallel_workers=1,
    )


def _require_released_training_gates() -> None:
    report_path = ROOT / "reports" / "second_stage_joint_repair_gate.md"
    if not report_path.is_file():
        raise RuntimeError(f"training gate report is missing: {report_path}")
    report_lines = {
        line.strip()
        for line in report_path.read_text(encoding="utf-8").splitlines()
    }
    required = (
        "PROPOSAL_COVERAGE_GATE = PASS",
        "CACHE_HYBRID_QUALITY_GATE = PASS",
        "CACHE_HYBRID_EFFICIENCY_GATE = PASS",
        "CACHE_CANDIDATE_TEACHER_SCALE_GATE = PASS",
        "PPO_REGRESSION_SMOKE = PASS",
        "LR_RUNNER_INITIALIZATION_GATE = PASS",
        "READY_FOR_FORMAL_TRAINING_STAGE = YES",
    )
    missing = [line for line in required if line not in report_lines]
    if missing:
        raise RuntimeError(
            "one-epoch training is blocked by reports/second_stage_joint_repair_gate.md; "
            "required passing fields are missing: " + ", ".join(missing)
        )


def run_one_epoch_runner(
    *,
    run_name: str,
    run_dir: str,
    lr_label: str,
    actor_lr: float,
) -> None:
    parser = argparse.ArgumentParser(description=f"Fresh one-epoch run: {run_name}")
    parser.add_argument(
        "--verify-initialization",
        action="store_true",
        help="hash fresh model initialization without rollout, optimizer step, or run creation",
    )
    args = parser.parse_args()
    output_dir = ROOT / run_dir

    if args.verify_initialization:
        trainer = _initialize_trainer(actor_lr)
        metadata = trainer.learning_rate_metadata()
        hashes = _parameter_hashes(trainer)
        branch_rates = {
            key: metadata[key]
            for key in (
                "cache_learning_rate",
                "uav_operation_learning_rate",
                "uav_candidate_learning_rate",
                "effective_critic_learning_rate",
            )
        }
        branch_rates["effective_actor_learning_rate"] = metadata[
            "effective_actor_learning_rate"
        ]
        actor_rates = [
            float(branch_rates[key])
            for key in (
                "effective_actor_learning_rate",
                "cache_learning_rate",
                "uav_operation_learning_rate",
                "uav_candidate_learning_rate",
            )
        ]
        if any(rate != float(actor_lr) for rate in actor_rates):
            raise RuntimeError(f"actor branches do not share requested LR: {actor_rates}")
        if float(branch_rates["effective_critic_learning_rate"]) != CRITIC_LR:
            raise RuntimeError("critic LR differs from the required 1e-3")
        steps = {
            "optimizer_steps": int(trainer.optimizer_steps),
            "actor_optimizer_steps": int(trainer.actor_optimizer_steps),
            "critic_optimizer_steps": int(trainer.critic_optimizer_steps),
            "cache_optimizer_steps": int(trainer.cache_optimizer_steps),
            "uav_operation_optimizer_steps": int(
                trainer.uav_operation_optimizer_steps
            ),
            "uav_candidate_optimizer_steps": int(
                trainer.uav_candidate_optimizer_steps
            ),
        }
        if any(value != 0 for value in steps.values()):
            raise RuntimeError(f"initialization verification took an optimizer step: {steps}")
        if output_dir.exists():
            raise RuntimeError(
                "verification run directory already exists; no files were changed: "
                f"{output_dir}"
            )
        print(
            json.dumps(
                {
                    "gate": "LR_RUNNER_INITIALIZATION_GATE",
                    "status": "PASS",
                    "run_name": run_name,
                    "run_dir": str(output_dir),
                    "lr_label": lr_label,
                    "actor_lr": float(actor_lr),
                    "critic_lr": CRITIC_LR,
                    "branch_learning_rates": branch_rates,
                    "optimizer_steps": steps,
                    "run_dir_created": False,
                    **hashes,
                },
                sort_keys=True,
            )
        )
        return

    if output_dir.exists():
        raise RuntimeError(
            "one-epoch output directory already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    if run_name != "one_epoch_actor_lr_5e_minus_4" or float(actor_lr) != 5.0e-4:
        raise RuntimeError(
            "this task authorizes training only with the 5e-4 actor-LR runner; "
            "other runners are initialization-verification only"
        )
    _require_released_training_gates()
    run_training(
        str(CONFIG_PATH),
        1,
        str(output_dir),
        episode_pool=str(EPISODE_POOL),
        actor_learning_rate_override=float(actor_lr),
        critic_learning_rate_override=CRITIC_LR,
        exact_q_parallel_workers=1,
        resume=False,
        verbose=True,
    )
