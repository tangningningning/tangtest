from __future__ import annotations

import hashlib
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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15c2_trigger_normalization_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15c2_trigger_normalization_gate.json"
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
NORMALIZATION_SOURCE = PROJECT_ROOT / "reports" / "critic_independent_trigger_target_probe.json"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 2
MACRO_ACTOR_LR = 1.0e-3
TARGET_KL = 0.02
HARD_KL = 0.05
BACKTRACKING_SCALES = [1.0, 0.25, 0.0625, 0.015625]
MAX_GRAD_NORM = 0.5
LOW_VARIANCE_THRESHOLD = 1.0e-6
TRIGGER_INPUT_CLIP = [-10.0, 10.0]
EXPECTED_NORMALIZATION_SOURCE_SHA256 = "1be40fb05ffce97bf3fe58085c1229661feb8b5c92f4556010b4ab5396324af2"
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15c2_trigger_normalization_gate.py"
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
        raise ValueError(f"C2 run directory must be a direct child of runs/: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {resolved}")


def _resume_interrupted_run(run_dir: Path) -> bool:
    """Resume only the verified one-episode prefix of this exact C2 gate."""

    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        raise FileExistsError(f"C2 run path is not a directory: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "latest_checkpoint.pt"
    metrics_path = run_dir / "episode_metrics.jsonl"
    required = (manifest_path, summary_path, checkpoint_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(
            f"refusing to resume an incomplete/unverified C2 run; missing {missing}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm_id": "V15-C2_TRIGGER_NORMALIZATION",
        "architecture_version": 22,
        "checkpoint_version": 46,
        "target_episodes": TARGET_EPISODES,
        "episode_seed_start": EPISODE_SEED_START,
        "stage1_trigger_only": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise FileExistsError("refusing to resume a run with mismatched C2 manifest")
    normalizer_metadata = manifest.get("trigger_observation_normalization_metadata", {})
    if (
        normalizer_metadata.get("source_sha256")
        != EXPECTED_NORMALIZATION_SOURCE_SHA256
        or normalizer_metadata.get("scale_rule")
        != "unit_scale_for_low_variance"
    ):
        raise FileExistsError("refusing to resume a run with mismatched C2 normalizer")
    if any(
        summary.get(key) != value
        for key, value in (
            ("algorithm_id", expected["algorithm_id"]),
            ("architecture_version", expected["architecture_version"]),
            ("checkpoint_version", expected["checkpoint_version"]),
            ("target_episodes", TARGET_EPISODES),
            ("stage1_trigger_only", True),
            ("completed_episodes", 1),
        )
    ):
        raise FileExistsError("refusing to resume an unexpected C2 run summary")
    episode_numbers = [
        int(record["episode"])
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
        if record.get("record_type") == "episode"
    ]
    if episode_numbers != [1]:
        raise FileExistsError(
            f"refusing to resume C2 with episode records {episode_numbers}; expected [1]"
        )
    return True


def _load_training_only_normalizer() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not NORMALIZATION_SOURCE.is_file() or not V15C_NORMALIZATION_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing frozen normalization source/checkpoint: {NORMALIZATION_SOURCE}, "
            f"{V15C_NORMALIZATION_CHECKPOINT}"
        )
    source_bytes = NORMALIZATION_SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_NORMALIZATION_SOURCE_SHA256:
        raise ValueError(
            "The training-only normalization source changed; refusing to recompute "
            "statistics from an unexpected artifact."
        )
    source_payload = json.loads(source_bytes.decode("utf-8"))
    states = source_payload.get("states")
    if not isinstance(states, list) or len(states) != 200:
        raise ValueError("Normalization source must contain exactly 200 frozen states.")
    expected_seeds = set(range(20041, 20051))
    observed_seeds = {int(state["episode_seed"]) for state in states}
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Normalization source seed set mismatch: {sorted(observed_seeds)}"
        )
    observations = np.asarray(
        [state["observation_400d"] for state in states], dtype=np.float64
    )
    if observations.shape != (200, 400) or not np.isfinite(observations).all():
        raise ValueError("Normalization source must contain finite [200,400] observations.")
    raw_mean_64 = observations.mean(axis=0, dtype=np.float64)
    raw_std_64 = observations.std(axis=0, ddof=0, dtype=np.float64)
    raw_mean = raw_mean_64.astype(np.float32)
    raw_std = raw_std_64.astype(np.float32)
    low_variance_mask = raw_std_64 <= LOW_VARIANCE_THRESHOLD
    scale = np.where(low_variance_mask, 1.0, raw_std_64).astype(np.float32)

    checkpoint = torch.load(
        V15C_NORMALIZATION_CHECKPOINT, map_location="cpu", weights_only=False
    )
    previous = checkpoint.get("trigger_observation_normalization")
    if not isinstance(previous, dict) or previous.get("active") is not True:
        raise ValueError("V15-C checkpoint has no active Trigger normalization metadata.")
    previous_metadata = dict(previous.get("metadata", {}))
    if previous_metadata.get("source_sha256") != source_sha256:
        raise ValueError("V15-C checkpoint and frozen training-only source disagree.")
    previous_mean = np.asarray(previous.get("mean"), dtype=np.float32).reshape(-1)
    previous_std = np.asarray(previous.get("std"), dtype=np.float32).reshape(-1)
    if (
        previous_mean.shape != (400,)
        or previous_std.shape != (400,)
        or not np.array_equal(previous_mean, raw_mean)
        or not np.array_equal(
            previous_std,
            np.maximum(raw_std_64, LOW_VARIANCE_THRESHOLD).astype(np.float32),
        )
    ):
        raise ValueError(
            "Reconstructed raw statistics do not reproduce the saved V15-C normalizer."
        )
    if not np.isfinite(raw_mean).all() or not np.isfinite(raw_std).all():
        raise FloatingPointError("Raw normalization statistics contain NaN/Inf.")
    metadata = {
        "normalization_version": "V15-C2_RAW_STD_UNIT_SCALE_CLIP10",
        "source_path": str(NORMALIZATION_SOURCE.resolve()),
        "source_sha256": source_sha256,
        "source_training_episode_seeds": sorted(expected_seeds),
        "sample_count": int(observations.shape[0]),
        "feature_dim": int(observations.shape[1]),
        "standard_deviation_ddof": 0,
        "low_variance_threshold": LOW_VARIANCE_THRESHOLD,
        "low_variance_feature_count": int(low_variance_mask.sum()),
        "scale_rule": "unit_scale_for_low_variance",
        "low_variance_scale": 1.0,
        "input_clip": list(TRIGGER_INPUT_CLIP),
        "normalization_scope": "Trigger input only; Candidate/Duration/Critic/Micro inputs unchanged",
    }
    normalizer = {
        "mean": raw_mean.copy(),
        "std": raw_std.copy(),
        "raw_mean": raw_mean.copy(),
        "raw_std": raw_std.copy(),
        "scale": scale.copy(),
        "metadata": metadata,
    }
    source = {
        "source_artifact": str(NORMALIZATION_SOURCE.resolve()),
        "source_sha256": source_sha256,
        "source_checkpoint": str(V15C_NORMALIZATION_CHECKPOINT.resolve()),
        "source_checkpoint_episode": int(
            checkpoint.get("training_counters", {}).get("completed_episodes", -1)
        ),
        "source_checkpoint_version": int(checkpoint.get("checkpoint_version", -1)),
        "source_v15c_normalizer_sha256": previous_metadata.get("source_sha256"),
        "feature_count": 400,
        "low_variance_threshold": LOW_VARIANCE_THRESHOLD,
        "low_variance_feature_count": int(low_variance_mask.sum()),
        "raw_mean": raw_mean.astype(float).tolist(),
        "raw_std": raw_std.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "stats_recomputed_from_frozen_training_only_observations": True,
        "teacher_recomputed": False,
        "source_episode_seeds": sorted(expected_seeds),
    }
    return normalizer, source


def _normalization_unit_and_clip_check(
    trainer: Any, normalization: dict[str, Any]
) -> dict[str, Any]:
    import torch

    test_mean = np.zeros(400, dtype=np.float32)
    test_std = np.ones(400, dtype=np.float32)
    test_mean[1] = 1.0
    test_std[0] = 0.0
    test_std[1] = 0.0
    test_metadata = {
        "normalization_version": "V15-C2_RAW_STD_UNIT_SCALE_CLIP10_TEST",
        "low_variance_threshold": LOW_VARIANCE_THRESHOLD,
        "low_variance_feature_count": 2,
        "scale_rule": "unit_scale_for_low_variance",
        "input_clip": list(TRIGGER_INPUT_CLIP),
    }
    trainer.policy.set_trigger_observation_normalization(
        test_mean, test_std, metadata=test_metadata
    )
    values = np.zeros(400, dtype=np.float32)
    values[0] = 1.0  # training constant 0 -> realization value 1
    values[1] = 0.0  # training constant 1 -> realization value 0
    values[2] = 25.0
    values[3] = -25.0
    values_tensor = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
    preclip = trainer.policy._preclip_normalized_trigger_observations(values_tensor)
    normalized = trainer.policy._normalized_trigger_observations(values_tensor)
    unit_zero_result = float(preclip[0, 0].item())
    unit_one_result = float(preclip[0, 1].item())
    preclip_max = float(preclip.abs().max().item())
    postclip_max = float(normalized.abs().max().item())
    if not (
        math.isclose(unit_zero_result, 1.0, rel_tol=0.0, abs_tol=1.0e-7)
        and math.isclose(unit_one_result, -1.0, rel_tol=0.0, abs_tol=1.0e-7)
        and postclip_max <= 10.0
        and float(normalized[0, 2].item()) == 10.0
        and float(normalized[0, 3].item()) == -10.0
        and bool(torch.isfinite(normalized).all().item())
    ):
        raise AssertionError(
            "V15-C2 normalization unit/clip check failed: "
            f"constant0={unit_zero_result}, constant1={unit_one_result}, "
            f"preclip_max={preclip_max}, postclip_max={postclip_max}"
        )
    trainer.policy.set_trigger_observation_normalization(
        normalization["raw_mean"],
        normalization["raw_std"],
        metadata=normalization["metadata"],
    )
    return {
        "pass": True,
        "constant_zero_feature_new_value_1_normalized": unit_zero_result,
        "constant_one_feature_new_value_0_normalized": unit_one_result,
        "preclip_abs_max": preclip_max,
        "postclip_abs_max": postclip_max,
        "clip_range": list(TRIGGER_INPUT_CLIP),
        "nan_inf_free": bool(torch.isfinite(normalized).all().item()),
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
        raise AssertionError(f"Fresh C2 Trigger initialization is not P_UPDATE=.5: {p_update}")
    normalization_finite = (
        np.isfinite(normalization["raw_mean"]).all()
        and np.isfinite(normalization["raw_std"]).all()
        and np.isfinite(normalization["scale"]).all()
    )
    if not normalization_finite:
        raise FloatingPointError("Trigger normalizer contains NaN/Inf.")
    normalization_check = _normalization_unit_and_clip_check(trainer, normalization)
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
    configured_grad_norm = float(trainer.trigger_max_grad_norm)
    if not math.isclose(
        configured_grad_norm, MAX_GRAD_NORM, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise AssertionError(
            f"Formal Trigger max_grad_norm is {configured_grad_norm}, expected 0.5."
        )
    return {
        "pass": True,
        "v14b_warmstart": transplant,
        "fresh_trigger_initial_p_update": p_update,
        "trigger_lr": MACRO_ACTOR_LR,
        "normalization_finite": bool(normalization_finite),
        "normalization_recomputed_from_frozen_training_only_observations": True,
        "teacher_recomputed": False,
        "low_variance_feature_count": int(
            normalization["metadata"]["low_variance_feature_count"]
        ),
        "trigger_input_clip": list(TRIGGER_INPUT_CLIP),
        "trigger_grad_clip_active": True,
        "trigger_max_grad_norm": configured_grad_norm,
        "trigger_optimizer_vs_frozen_network_overlap": 0,
        "normalization_unit_and_clip_check": normalization_check,
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

    accepted_attempts: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    scale_pairs: list[tuple[float, float]] = []
    for record in records:
        for epoch in record.get("trigger_epoch_records", []):
            attempts = epoch.get("attempts", [])
            for attempt in attempts:
                entry = {
                    "episode": int(record.get("episode", -1)),
                    "epoch": int(epoch.get("epoch", -1)),
                    "step_scale": float(attempt.get("step_scale", 0.0)),
                    "kl_mean": float(attempt.get("kl_mean", 0.0)),
                    "kl_max": float(attempt.get("kl_max", 0.0)),
                    "accepted": bool(attempt.get("accepted")),
                }
                (accepted_attempts if entry["accepted"] else rejected_attempts).append(entry)
            scale_pairs.extend(
                (float(left["kl_mean"]), float(right["kl_mean"]))
                for left, right in zip(attempts, attempts[1:])
            )

    backtracking_response = bool(
        scale_pairs
        and all(left >= right for left, right in scale_pairs)
        and any(left > right + 1.0e-12 for left, right in scale_pairs)
    )
    accepted_kl_mean_max = max(
        (item["kl_mean"] for item in accepted_attempts), default=0.0
    )
    accepted_kl_per_state_max = max(
        (item["kl_max"] for item in accepted_attempts), default=0.0
    )
    rejected_kl_mean_max = max(
        (item["kl_mean"] for item in rejected_attempts), default=0.0
    )
    rejected_kl_per_state_max = max(
        (item["kl_max"] for item in rejected_attempts), default=0.0
    )

    def episode_summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "episode": record.get("episode"),
            "episode_seed": record.get("episode_seed"),
            "preclip_normalized_obs_abs_max": record.get(
                "trigger_preclip_normalized_obs_abs_max"
            ),
            "postclip_normalized_obs_abs_max": record.get(
                "trigger_postclip_normalized_obs_abs_max",
                record.get("trigger_normalized_obs_abs_max"),
            ),
            "normalized_obs_p99_abs": record.get("trigger_normalized_obs_p99_abs"),
            "normalized_obs_mean_abs": record.get("trigger_normalized_obs_mean_abs"),
            "clipped_value_fraction": record.get(
                "trigger_normalized_obs_clipped_value_fraction"
            ),
            "preclip_grad_norm": record.get("trigger_preclip_grad_norm"),
            "postclip_grad_norm": record.get("trigger_postclip_grad_norm"),
            "accepted_epoch_count": int(
                record.get("trigger_epoch_accepted_count", 0)
            ),
            "rejected_epoch_count": int(
                record.get("trigger_epoch_rejected_count", 0)
            ),
            "rollback_count": int(record.get("trigger_rollback_count_this_update", 0)),
            "accepted_kl_mean_max": max(
                (
                    float(attempt["kl_mean"])
                    for epoch in record.get("trigger_epoch_records", [])
                    for attempt in epoch.get("attempts", [])
                    if attempt.get("accepted")
                ),
                default=0.0,
            ),
            "accepted_kl_per_state_max": max(
                (
                    float(attempt["kl_max"])
                    for epoch in record.get("trigger_epoch_records", [])
                    for attempt in epoch.get("attempts", [])
                    if attempt.get("accepted")
                ),
                default=0.0,
            ),
            "rejected_kl_mean_max": max(
                (
                    float(attempt["kl_mean"])
                    for epoch in record.get("trigger_epoch_records", [])
                    for attempt in epoch.get("attempts", [])
                    if not attempt.get("accepted")
                ),
                default=0.0,
            ),
            "rejected_kl_per_state_max": max(
                (
                    float(attempt["kl_max"])
                    for epoch in record.get("trigger_epoch_records", [])
                    for attempt in epoch.get("attempts", [])
                    if not attempt.get("accepted")
                ),
                default=0.0,
            ),
            "p_update_after": record.get("trigger_p_update_after"),
            "p_update_state_std_after": record.get("trigger_p_update_state_std"),
            "entropy_after": record.get("trigger_entropy"),
            "policy_delta_spearman": record.get("trigger_policy_delta_spearman"),
            "preupdate_p_update_mean": record.get("trigger_p_update_before"),
            "preupdate_p_update_std": record.get("trigger_p_update_std_before"),
            "preupdate_entropy": record.get("trigger_entropy_before"),
            "epoch_attempts": [
                {
                    "epoch": int(epoch.get("epoch", -1)),
                    "accepted": bool(epoch.get("accepted")),
                    "attempts": epoch.get("attempts", []),
                }
                for epoch in record.get("trigger_epoch_records", [])
            ],
        }

    summaries = [episode_summary(record) for record in records]
    ep1 = summaries[0] if summaries else {}
    ep2 = summaries[1] if len(summaries) > 1 else {}
    preclip_maxima = [
        item["preclip_normalized_obs_abs_max"]
        for item in summaries
        if item["preclip_normalized_obs_abs_max"] is not None
    ]
    postclip_maxima = [
        item["postclip_normalized_obs_abs_max"]
        for item in summaries
        if item["postclip_normalized_obs_abs_max"] is not None
    ]
    input_safe = (
        len(records) == TARGET_EPISODES
        and all(float(value) <= 10.0 for value in postclip_maxima)
        and all(
            math.isfinite(float(value)) and float(value) < 1.0e6
            for value in preclip_maxima
        )
    )
    ep2_gate = (
        ep2.get("preupdate_p_update_mean") is not None
        and ep2.get("preupdate_p_update_std") is not None
        and ep2.get("preupdate_entropy") is not None
        and 0.05 < float(ep2["preupdate_p_update_mean"]) < 0.95
        and float(ep2["preupdate_p_update_std"]) > 0.0
        and float(ep2["preupdate_entropy"]) > 0.0
    )
    ep1_learned_state_dependence = (
        ep1.get("p_update_state_std_after") is not None
        and float(ep1["p_update_state_std_after"]) > 0.0
    )
    all_accepted_kl_within_hard = all(
        item["kl_mean"] <= HARD_KL for item in accepted_attempts
    )
    successful = (
        input_safe
        and int(ep1.get("accepted_epoch_count", 0)) >= 1
        and all_accepted_kl_within_hard
        and ep1_learned_state_dependence
        and ep2_gate
    )
    root_cause_supported = (
        "YES"
        if successful
        else "PARTIAL"
        if input_safe
        else "NO"
    )
    if successful:
        next_step = (
            "Restore V15-C Stage-1 and run the full 30 episodes; that longer run was "
            "not started in this 2-episode gate."
        )
    elif input_safe and int(ep1.get("accepted_epoch_count", 0)) == 0:
        next_step = (
            "Input scale is repaired, but episode 1 accepted no Trigger step. Inspect "
            "optimizer step size/LR next without changing it in this run."
        )
    elif input_safe:
        next_step = (
            "Input scale is repaired, but state-dependence or the episode 2 gate did "
            "not pass. Inspect the recorded Trigger diagnostics before a longer run."
        )
    else:
        next_step = (
            "Inspect raw statistics and Trigger preclip diagnostics before any longer "
            "Stage-1 run."
        )

    finite_keys = (
        "trigger_kl_mean",
        "trigger_kl_max",
        "trigger_preclip_normalized_obs_abs_max",
        "trigger_postclip_normalized_obs_abs_max",
        "trigger_normalized_obs_p99_abs",
        "trigger_normalized_obs_mean_abs",
        "trigger_normalized_obs_clipped_value_fraction",
        "trigger_preclip_grad_norm",
        "trigger_postclip_grad_norm",
    )
    finite_records = all(
        math.isfinite(float(record[key]))
        for record in records
        for key in finite_keys
        if record.get(key) is not None
    )
    completed = len(records) == TARGET_EPISODES
    implementation_complete = bool(
        completed
        and preflight.get("pass")
        and finite_records
        and train_result.get("completed_episodes") == TARGET_EPISODES
    )
    return {
        "experiment": "V15-C2 Trigger normalization safety gate",
        "status": "COMPLETED" if completed else "INCOMPLETE",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "downstream_warmstart_from_v14b": True,
        "trigger_reinitialized": True,
        "trigger_learning_rate": MACRO_ACTOR_LR,
        "trigger_max_grad_norm": MAX_GRAD_NORM,
        "trigger_target_kl": TARGET_KL,
        "trigger_hard_kl": HARD_KL,
        "trigger_backtracking_step_scales": BACKTRACKING_SCALES,
        "trigger_transactional_kl_rollback_active": True,
        "trigger_observation_normalization_source": normalization_source,
        "trigger_observation_normalization_recomputed_from_frozen_source": True,
        "teacher_recomputed": False,
        "preflight": preflight,
        "backtracking_attempt_kl": accepted_attempts + rejected_attempts,
        "backtracking_kl_scale_pairs": [
            {"larger_scale_kl_mean": left, "smaller_scale_kl_mean": right}
            for left, right in scale_pairs
        ],
        "accepted_kl_mean_max": accepted_kl_mean_max,
        "accepted_kl_per_state_max": accepted_kl_per_state_max,
        "rejected_kl_mean_max": rejected_kl_mean_max,
        "rejected_kl_per_state_max": rejected_kl_per_state_max,
        "all_accepted_kl_within_hard_limit": all_accepted_kl_within_hard,
        "input_scale_safe": input_safe,
        "episode1": ep1,
        "episode2": ep2,
        "episodes": summaries,
        "nan_inf_free": bool(finite_records),
        "IMPLEMENTATION_COMPLETE": "YES" if implementation_complete else "NO",
        "LOW_VARIANCE_FEATURE_COUNT": normalization_source["low_variance_feature_count"],
        "TRIGGER_INPUT_CLIP": list(TRIGGER_INPUT_CLIP),
        "EPISODES_COMPLETED": len(records),
        "EP1_PRECLIP_NORMALIZED_ABS_MAX": ep1.get("preclip_normalized_obs_abs_max"),
        "EP1_POSTCLIP_NORMALIZED_ABS_MAX": ep1.get("postclip_normalized_obs_abs_max"),
        "EP2_PRECLIP_NORMALIZED_ABS_MAX": ep2.get("preclip_normalized_obs_abs_max"),
        "EP2_POSTCLIP_NORMALIZED_ABS_MAX": ep2.get("postclip_normalized_obs_abs_max"),
        "EP1_PRECLIP_GRAD_NORM": ep1.get("preclip_grad_norm"),
        "EP1_POSTCLIP_GRAD_NORM": ep1.get("postclip_grad_norm"),
        "EP1_ACCEPTED_EPOCH_COUNT": ep1.get("accepted_epoch_count", 0),
        "EP1_REJECTED_EPOCH_COUNT": ep1.get("rejected_epoch_count", 0),
        "EP1_ROLLBACK_COUNT": ep1.get("rollback_count", 0),
        "EP1_ACCEPTED_KL_MAX": ep1.get("accepted_kl_mean_max", 0.0),
        "EP1_ACCEPTED_KL_PER_STATE_MAX": ep1.get("accepted_kl_per_state_max", 0.0),
        "EP1_REJECTED_KL_MAX": ep1.get("rejected_kl_mean_max", 0.0),
        "EP1_REJECTED_KL_PER_STATE_MAX": ep1.get("rejected_kl_per_state_max", 0.0),
        "BACKTRACKING_KL_SCALE_RESPONSE": "YES" if backtracking_response else "NO",
        "EP1_P_UPDATE_AFTER": ep1.get("p_update_after"),
        "EP1_P_UPDATE_STATE_STD_AFTER": ep1.get("p_update_state_std_after"),
        "EP2_PREUPDATE_P_UPDATE_MEAN": ep2.get("preupdate_p_update_mean"),
        "EP2_PREUPDATE_P_UPDATE_STD": ep2.get("preupdate_p_update_std"),
        "EP2_PREUPDATE_ENTROPY": ep2.get("preupdate_entropy"),
        "NORMALIZATION_ROOT_CAUSE_SUPPORTED": root_cause_supported,
        "V15C_NUMERICAL_STABILITY_FIXED": "YES" if successful else "NO",
        "NEXT_STEP": next_step,
        "run_training_result": train_result,
    }


def main() -> int:
    resume_existing = _resume_interrupted_run(RUN_DIR)
    if not resume_existing:
        _assert_fresh_run_dir(RUN_DIR)
    input_files = [
        CONFIG_PATH,
        V14B_CHECKPOINT,
        V15C_NORMALIZATION_CHECKPOINT,
        NORMALIZATION_SOURCE,
    ]
    missing = [str(path) for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V15-C2 input missing: {missing}")
    episodes = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing_episodes = [str(path) for path in episodes if not path.is_dir()]
    if missing_episodes:
        raise FileNotFoundError(f"C2 requires existing episode realizations: {missing_episodes}")
    normalization, normalization_source = _load_training_only_normalizer()
    preflight = _preflight(normalization)
    _write_json(
        REPORT_PATH,
        {
            "experiment": "V15-C2 Trigger normalization safety gate",
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
    _write_json(RUN_DIR / "v15c2_trigger_normalization_gate_report.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "IMPLEMENTATION_COMPLETE",
                    "LOW_VARIANCE_FEATURE_COUNT",
                    "EPISODES_COMPLETED",
                    "EP1_PRECLIP_NORMALIZED_ABS_MAX",
                    "EP1_POSTCLIP_NORMALIZED_ABS_MAX",
                    "EP1_ACCEPTED_EPOCH_COUNT",
                    "EP1_ROLLBACK_COUNT",
                    "BACKTRACKING_KL_SCALE_RESPONSE",
                    "EP2_PREUPDATE_P_UPDATE_MEAN",
                    "EP2_PREUPDATE_P_UPDATE_STD",
                    "NORMALIZATION_ROOT_CAUSE_SUPPORTED",
                    "V15C_NUMERICAL_STABILITY_FIXED",
                    "NEXT_STEP",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
