from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "proposed" / "system_config.toml"
EPISODE_POOL = PROJECT_ROOT / "environment_data" / "proposed_joint_training"
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15c1_trigger_trust_region_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15c1_trigger_trust_region_gate.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
V15C_NORMALIZATION_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v15c_state_dependent_trigger_gate"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 2
MACRO_ACTOR_LR = 1.0e-3
TARGET_KL = 0.02
HARD_KL = 0.05
BACKTRACKING_SCALES = [1.0, 0.25, 0.0625, 0.015625]
MAX_GRAD_NORM = 0.5
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15c1_trigger_trust_region_gate.py"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _assert_fresh_run_dir(run_dir: Path) -> None:
    resolved = run_dir.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if resolved.parent != runs_root:
        raise ValueError(f"C1 run directory must be a direct child of runs/: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {resolved}")


def _resume_interrupted_run(run_dir: Path) -> bool:
    """Resume only the verified one-episode prefix of this exact two-episode gate."""

    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        raise FileExistsError(f"C1 run path is not a directory: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    metrics_path = run_dir / "episode_metrics.jsonl"
    required = (manifest_path, summary_path, checkpoint_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(
            f"refusing to resume an incomplete/unverified C1 run; missing {missing}"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    expected_manifest = {
        "algorithm_id": "V15-C1_TRIGGER_TRUST_REGION",
        "architecture_version": 22,
        "checkpoint_version": 45,
        "target_episodes": TARGET_EPISODES,
        "episode_seed_start": EPISODE_SEED_START,
        "stage1_trigger_only": True,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise FileExistsError(
            f"refusing to resume a run that does not match C1: {mismatches}"
        )
    if (
        summary.get("algorithm_id") != expected_manifest["algorithm_id"]
        or summary.get("architecture_version") != expected_manifest["architecture_version"]
        or summary.get("checkpoint_version") != expected_manifest["checkpoint_version"]
        or summary.get("target_episodes") != TARGET_EPISODES
        or summary.get("stage1_trigger_only") is not True
        or summary.get("completed_episodes") != 1
    ):
        raise FileExistsError("refusing to resume an unexpected C1 run summary")
    with metrics_path.open("r", encoding="utf-8") as stream:
        episode_numbers = [
            int(record["episode"])
            for line in stream
            if line.strip()
            for record in [json.loads(line)]
            if record.get("record_type") == "episode"
        ]
    if episode_numbers != [1]:
        raise FileExistsError(
            f"refusing to resume C1 with episode records {episode_numbers}; expected [1]"
        )
    return True


def _load_existing_normalizer() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not V15C_NORMALIZATION_CHECKPOINT.is_file():
        raise FileNotFoundError(V15C_NORMALIZATION_CHECKPOINT)
    checkpoint = torch.load(
        V15C_NORMALIZATION_CHECKPOINT, map_location="cpu", weights_only=False
    )
    payload = checkpoint.get("trigger_observation_normalization")
    if not isinstance(payload, dict) or payload.get("active") is not True:
        raise ValueError("Previous V15-C checkpoint has no active Trigger normalizer.")
    mean = np.asarray(payload.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(payload.get("std"), dtype=np.float32).reshape(-1)
    if (
        mean.shape != (400,)
        or std.shape != (400,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < np.float32(1.0e-6))
    ):
        raise ValueError("Saved V15-C Trigger normalization arrays are invalid.")
    metadata = dict(payload.get("metadata", {}))
    floor_indices = np.flatnonzero(std <= np.float32(1.0e-6))
    expected_floor_count = int(metadata.get("standard_deviation_floor_feature_count", -1))
    if len(floor_indices) != 73 or expected_floor_count != 73:
        raise ValueError(
            "C1 must preserve the existing 73-feature std_floor=1e-6 normalizer."
        )
    normalizer = {"mean": mean.copy(), "std": std.copy(), "metadata": metadata}
    source = {
        "source_checkpoint": str(V15C_NORMALIZATION_CHECKPOINT.resolve()),
        "source_checkpoint_episode": int(
            checkpoint.get("training_counters", {}).get("completed_episodes", -1)
        ),
        "source_checkpoint_version": int(checkpoint.get("checkpoint_version", -1)),
        "source_v14b_sha256": metadata.get("source_sha256"),
        "feature_count": 400,
        "std_floor": 1.0e-6,
        "std_floor_feature_count": int(len(floor_indices)),
        "normalizer_recomputed": False,
        "normalizer_sha_source": metadata.get("source_sha256"),
    }
    return normalizer, source


def _nested_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _transaction_rollback_synthetic_check() -> dict[str, Any]:
    import torch

    from ppo_simple.trainer import _transactional_optimizer_update

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    warmup_loss = model(torch.ones(1, 3)).square().sum()
    warmup_loss.backward()
    optimizer.step()
    pre_model = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    pre_optimizer = optimizer.state_dict()
    initial_optimizer_state = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    initial_optimizer_state.load_state_dict(pre_optimizer)
    pre_optimizer_copy = initial_optimizer_state.state_dict()

    def step_fn() -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        loss = -model(torch.full((1, 3), 100.0)).sum()
        loss.backward()
        pre_clip = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM).item()
        )
        return {"pre_clip_grad_norm": pre_clip}

    transaction = _transactional_optimizer_update(
        model,
        optimizer,
        step_fn=step_fn,
        kl_fn=lambda: (0.050001, 0.2),
        hard_kl=HARD_KL,
        step_scales=tuple(BACKTRACKING_SCALES),
    )
    model_restored = all(
        torch.equal(model.state_dict()[key], value) for key, value in pre_model.items()
    )
    optimizer_restored = _nested_equal(optimizer.state_dict(), pre_optimizer_copy)
    if transaction["accepted"] or transaction["rollback_count"] != 4:
        raise AssertionError("Forced over-limit KL did not reject and roll back all attempts.")
    if not model_restored or not optimizer_restored:
        raise AssertionError("Transactional rollback failed bitwise model/optimizer restoration.")
    return {
        "pass": True,
        "forced_mean_kl": 0.050001,
        "attempt_count": int(transaction["attempt_count"]),
        "rollback_count": int(transaction["rollback_count"]),
        "model_parameters_bitwise_restored": model_restored,
        "optimizer_state_bitwise_restored": optimizer_restored,
        "optimizer_step_attempted_before_rollback": True,
    }


def _gradient_clip_check(trainer: Any) -> dict[str, Any]:
    import torch

    configured = float(trainer.config["max_grad_norm"])
    if not math.isclose(configured, MAX_GRAD_NORM, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(f"Trigger max_grad_norm is {configured}, expected 0.5.")
    parameters = trainer.policy.macro_cache_trigger_actor_parameters()
    trainer.macro_cache_trigger_actor_optimizer.zero_grad(set_to_none=True)
    synthetic_loss = sum(parameter.sum() * 100.0 for parameter in parameters)
    synthetic_loss.backward()
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(parameters, configured).detach().cpu()
    )
    post_clip_norm = float(
        torch.sqrt(
            sum(parameter.grad.detach().square().sum() for parameter in parameters)
        ).cpu()
    )
    trainer.macro_cache_trigger_actor_optimizer.zero_grad(set_to_none=True)
    if not (pre_clip_norm > configured and post_clip_norm <= configured + 1.0e-6):
        raise AssertionError(
            f"Trigger gradient clipping check failed: pre={pre_clip_norm}, post={post_clip_norm}."
        )
    return {
        "pass": True,
        "trigger_grad_clip_active": True,
        "trigger_max_grad_norm": configured,
        "synthetic_pre_clip_grad_norm": pre_clip_norm,
        "synthetic_post_clip_grad_norm": post_clip_norm,
        "all_trigger_parameters_clipped": True,
    }


def _preflight(normalization: dict[str, Any]) -> dict[str, Any]:
    import torch

    from ppo_simple.interface import load_environment_context
    from ppo_simple.trainer import PPOTrainer, _seed_everything

    realization = EPISODE_POOL / f"episode_{EPISODE_SEED_START}"
    context = load_environment_context(str(CONFIG_PATH), realization_dir=realization)
    _seed_everything(BASE_SEED)
    trainer = PPOTrainer(
        context,
        macro_actor_learning_rate_override=MACRO_ACTOR_LR,
        stage1_trigger_only=True,
        trigger_observation_normalization=normalization,
    )
    transplant = trainer.load_v14b_downstream_checkpoint(
        V14B_CHECKPOINT, expected_sha256=V14B_SHA256
    )
    p_update = float(
        trainer.policy.cache_trigger_distribution(
            np.zeros(400, dtype=np.float32), np.asarray([True])
        ).probs[0, 1].item()
    )
    if not math.isclose(p_update, 0.5, rel_tol=0.0, abs_tol=1.0e-7):
        raise AssertionError(f"Fresh C1 Trigger initialization is not P_UPDATE=.5: {p_update}")
    normalization_finite = (
        np.isfinite(normalization["mean"]).all()
        and np.isfinite(normalization["std"]).all()
    )
    if not normalization_finite:
        raise FloatingPointError("Trigger normalizer contains NaN/Inf.")
    clip_check = _gradient_clip_check(trainer)
    transactional_check = _transaction_rollback_synthetic_check()
    trigger_ids = {
        id(parameter) for parameter in trainer.policy.macro_cache_trigger_actor_parameters()
    }
    frozen_parameter_ids = {
        id(parameter)
        for parameter in (
            *trainer.policy.macro_actor_parameters(),
            *trainer.policy.macro_critic_parameters(),
            *trainer.policy.long_value_critic_parameters(),
            *trainer.policy.micro_actor_parameters(),
            *trainer.policy.micro_critic_parameters(),
        )
    }
    if trigger_ids & frozen_parameter_ids:
        raise AssertionError("Trigger optimizer shares parameters with a frozen module.")
    optimizer_lrs = {
        float(group["lr"])
        for group in trainer.macro_cache_trigger_actor_optimizer.param_groups
    }
    if optimizer_lrs != {MACRO_ACTOR_LR}:
        raise AssertionError(f"Formal Trigger LR mismatch: {optimizer_lrs}")
    return {
        "pass": True,
        "v14b_warmstart": transplant,
        "fresh_trigger_initial_p_update": p_update,
        "trigger_lr": MACRO_ACTOR_LR,
        "normalization_finite": bool(normalization_finite),
        "normalization_recomputed": False,
        "trigger_optimizer_vs_frozen_network_overlap": 0,
        "transactional_rollback_synthetic_check": transactional_check,
        "gradient_clip_check": clip_check,
        "nan_inf_check": True,
    }


def _load_episode_metrics(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if item.get("record_type") == "episode":
                    records.append(item)
    return records


def _run_report(
    train_result: dict[str, Any],
    preflight: dict[str, Any],
    normalization_source: dict[str, Any],
) -> dict[str, Any]:
    records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
    first = records[0] if records else {}
    second = records[1] if len(records) > 1 else {}
    accepted_kl_values = []
    rejected_kl_values = []
    rollback_count = 0
    for record in records:
        for epoch in record.get("trigger_epoch_records", []):
            rollback_count += int(epoch.get("rollback_count", 0))
            for attempt in epoch.get("attempts", []):
                if attempt.get("accepted"):
                    accepted_kl_values.append(float(attempt["kl_mean"]))
                else:
                    rejected_kl_values.append(float(attempt["kl_mean"]))
    norm_max_ep1 = first.get("trigger_normalized_obs_abs_max")
    norm_max_ep2 = second.get("trigger_normalized_obs_abs_max")
    all_norm_max = [value for value in (norm_max_ep1, norm_max_ep2) if value is not None]
    normalization_risk = any(float(value) > 100.0 for value in all_norm_max)
    ep2_pre_p = second.get("trigger_p_update_before")
    ep2_pre_std = second.get("trigger_p_update_std_before")
    ep2_pre_entropy = second.get("trigger_entropy_before")
    catastrophic_fixed = (
        ep2_pre_p is not None
        and ep2_pre_std is not None
        and ep2_pre_entropy is not None
        and 0.05 < float(ep2_pre_p) < 0.95
        and float(ep2_pre_std) > 0.0
        and float(ep2_pre_entropy) > 0.0
    )
    all_accepted_kl_within_hard = all(
        value <= HARD_KL for value in accepted_kl_values
    )
    optimizer_overshoot_not_sufficient = (
        len(records) == TARGET_EPISODES
        and all_accepted_kl_within_hard
        and not catastrophic_fixed
    )
    if catastrophic_fixed:
        next_step = "The rollback repair passed the 2-episode safety gate; a 30-episode Stage-1 rerun is now allowed, but was not started here."
    elif optimizer_overshoot_not_sufficient and normalization_risk:
        next_step = "Repair the 73 std-floor features and cross-realization normalization before another Stage-1 run."
    elif optimizer_overshoot_not_sufficient:
        next_step = "Investigate remaining cross-realization Trigger behavior before another Stage-1 run."
    else:
        next_step = "Inspect rejected-step and normalized-input diagnostics before any longer Stage-1 run."
    finite_records = all(
        math.isfinite(float(record[key]))
        for record in records
        for key in (
            "trigger_kl_mean",
            "trigger_kl_max",
            "trigger_normalized_obs_abs_max",
            "trigger_normalized_obs_p99_abs",
            "trigger_normalized_obs_mean_abs",
        )
        if record.get(key) is not None
    )
    return {
        "experiment": "V15-C1 Trigger trust-region safety gate",
        "status": "COMPLETED" if len(records) == TARGET_EPISODES else "INCOMPLETE",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "episodes_completed": len(records),
        "downstream_warmstart_from_v14b": True,
        "trigger_reinitialized": True,
        "fresh_trigger_initial_p_update": preflight["fresh_trigger_initial_p_update"],
        "trigger_normalization_source": normalization_source,
        "trigger_normalization_recomputed": False,
        "trigger_transactional_kl_rollback_active": True,
        "trigger_target_kl": TARGET_KL,
        "trigger_hard_kl": HARD_KL,
        "trigger_backtracking_step_scales": BACKTRACKING_SCALES,
        "trigger_grad_clip_active": True,
        "trigger_max_grad_norm": MAX_GRAD_NORM,
        "formal_trigger_learning_rate": MACRO_ACTOR_LR,
        "non_trigger_optimizer_steps": int(
            (RUN_DIR / "run_summary.json").is_file()
            and json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
            .get("training_counters", {})
            .get("formal_non_trigger_optimizer_steps", 0)
        ),
        "preflight": preflight,
        "episode1_p_update_after": first.get("trigger_p_update_after"),
        "episode1_state_std_after": first.get("trigger_p_update_state_std"),
        "episode1_accepted_kl_max": max(
            (
                float(epoch.get("accepted_kl_mean", 0.0))
                for epoch in first.get("trigger_epoch_records", [])
                if epoch.get("accepted")
            ),
            default=0.0,
        ),
        "episode1_accepted_kl_per_state_max": max(
            (
                float(epoch.get("accepted_kl_max", 0.0))
                for epoch in first.get("trigger_epoch_records", [])
                if epoch.get("accepted")
            ),
            default=0.0,
        ),
        "episode1_rejected_kl_max": max(
            (
                float(attempt["kl_mean"])
                for epoch in first.get("trigger_epoch_records", [])
                for attempt in epoch.get("attempts", [])
                if not attempt.get("accepted")
            ),
            default=0.0,
        ),
        "episode1_rejected_kl_per_state_max": max(
            (
                float(attempt["kl_max"])
                for epoch in first.get("trigger_epoch_records", [])
                for attempt in epoch.get("attempts", [])
                if not attempt.get("accepted")
            ),
            default=0.0,
        ),
        "episode1_rollback_count": int(first.get("trigger_rollback_count_this_update", 0)),
        "episode2_preupdate_p_update_mean": ep2_pre_p,
        "episode2_preupdate_p_update_std": ep2_pre_std,
        "episode2_preupdate_entropy": ep2_pre_entropy,
        "normalized_obs_abs_max_episode1": norm_max_ep1,
        "normalized_obs_abs_p99_episode1": first.get("trigger_normalized_obs_p99_abs"),
        "normalized_obs_mean_abs_episode1": first.get("trigger_normalized_obs_mean_abs"),
        "features_abs_gt_10_count_episode1": first.get("trigger_features_abs_gt_10_count"),
        "features_abs_gt_100_count_episode1": first.get("trigger_features_abs_gt_100_count"),
        "normalized_obs_abs_max_episode2": norm_max_ep2,
        "normalized_obs_abs_p99_episode2": second.get("trigger_normalized_obs_p99_abs"),
        "normalized_obs_mean_abs_episode2": second.get("trigger_normalized_obs_mean_abs"),
        "features_abs_gt_10_count_episode2": second.get("trigger_features_abs_gt_10_count"),
        "features_abs_gt_100_count_episode2": second.get("trigger_features_abs_gt_100_count"),
        "std_floor_feature_count": normalization_source["std_floor_feature_count"],
        "std_floor_features_abs_max_episode1": first.get("trigger_std_floor_features_abs_max"),
        "std_floor_features_abs_gt_100_count_episode1": first.get("trigger_std_floor_features_abs_gt_100_count"),
        "std_floor_features_abs_max_episode2": second.get("trigger_std_floor_features_abs_max"),
        "std_floor_features_abs_gt_100_count_episode2": second.get("trigger_std_floor_features_abs_gt_100_count"),
        "normalization_numerical_risk": normalization_risk,
        "accepted_epoch_kl_values": accepted_kl_values,
        "rejected_epoch_kl_values": rejected_kl_values,
        "rollback_count_total": rollback_count,
        "all_accepted_kl_within_hard_limit": all_accepted_kl_within_hard,
        "catastrophic_cross_realization_collapse_fixed": catastrophic_fixed,
        "optimizer_overshoot_not_sufficient": optimizer_overshoot_not_sufficient,
        "nan_inf_free": bool(finite_records),
        "next_step": next_step,
        "per_episode_metrics": records,
        "run_training_result": train_result,
    }


def main() -> int:
    resume_existing = _resume_interrupted_run(RUN_DIR)
    if not resume_existing:
        _assert_fresh_run_dir(RUN_DIR)
    input_files = [CONFIG_PATH, V14B_CHECKPOINT, V15C_NORMALIZATION_CHECKPOINT]
    missing = [str(path) for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V15-C1 input missing: {missing}")
    episodes = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing_episodes = [str(path) for path in episodes if not path.is_dir()]
    if missing_episodes:
        raise FileNotFoundError(f"C1 requires existing episode realizations: {missing_episodes}")
    normalization, normalization_source = _load_existing_normalizer()
    preflight = _preflight(normalization)
    _write_json(
        REPORT_PATH,
        {
            "experiment": "V15-C1 Trigger trust-region safety gate",
            "status": "PREFLIGHT_PASSED",
            "formal_runner_path": str(Path(__file__).resolve()),
            "preflight": preflight,
            "trigger_normalization_source": normalization_source,
        },
    )

    from ppo_simple.trainer import run_training

    train_result = run_training(
        CONFIG_PATH,
        TARGET_EPISODES,
        RUN_DIR,
        episode_pool=EPISODE_POOL,
        episode_seed_start=EPISODE_SEED_START,
        generate_missing_episodes=False,
        macro_actor_learning_rate_override=MACRO_ACTOR_LR,
        seed=BASE_SEED,
        resume=resume_existing,
        early_gate_episodes=None,
        initial_checkpoint_path=None if resume_existing else V14B_CHECKPOINT,
        expected_initial_checkpoint_sha256=(
            None if resume_existing else V14B_SHA256
        ),
        stage1_trigger_only=True,
        trigger_observation_normalization=normalization,
    )
    result = _run_report(train_result, preflight, normalization_source)
    _write_json(REPORT_PATH, result)
    _write_json(RUN_DIR / "v15c1_trigger_trust_region_gate_report.json", result)
    print(
        json.dumps(
            {
                "episodes_completed": result["episodes_completed"],
                "ep1_p_update_after": result["episode1_p_update_after"],
                "ep1_state_std_after": result["episode1_state_std_after"],
                "ep1_accepted_kl_max": result["episode1_accepted_kl_max"],
                "ep1_rejected_kl_max": result["episode1_rejected_kl_max"],
                "ep1_rollback_count": result["episode1_rollback_count"],
                "ep2_preupdate_p_update_mean": result[
                    "episode2_preupdate_p_update_mean"
                ],
                "ep2_preupdate_p_update_std": result[
                    "episode2_preupdate_p_update_std"
                ],
                "ep2_preupdate_entropy": result["episode2_preupdate_entropy"],
                "normalization_numerical_risk": result[
                    "normalization_numerical_risk"
                ],
                "catastrophic_cross_realization_collapse_fixed": result[
                    "catastrophic_cross_realization_collapse_fixed"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
