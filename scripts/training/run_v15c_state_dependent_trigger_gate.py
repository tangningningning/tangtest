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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15c_state_dependent_trigger_gate"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15c_state_dependent_trigger_gate.json"
NORMALIZATION_SOURCE = PROJECT_ROOT / "reports" / "critic_independent_trigger_target_probe.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 30
MACRO_ACTOR_LR = 1.0e-3
V15B_STATE_STD = 0.00577998
STATE_STD_IMPROVEMENT_RATIO_MIN = 2.0
SPEARMAN_LAST10_TARGET = 0.20
TRIGGER_KL_MEAN_LIMIT = 0.05
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15c_state_dependent_trigger_gate.py"
)
EARLY_STOPPED_EXISTING_RUN = False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _assert_run_directory_is_safe(run_dir: Path) -> None:
    resolved = run_dir.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if resolved.parent != runs_root:
        raise ValueError(f"run directory must be a direct child of runs/: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {resolved}")


def _resume_own_partial_run_if_valid() -> bool:
    """Continue only a matching V15-C run directory with a saved checkpoint."""

    global EARLY_STOPPED_EXISTING_RUN

    if not RUN_DIR.exists() or not any(RUN_DIR.iterdir()):
        _assert_run_directory_is_safe(RUN_DIR)
        return False
    manifest_path = RUN_DIR / "run_manifest.json"
    summary_path = RUN_DIR / "run_summary.json"
    checkpoint_path = RUN_DIR / "checkpoints" / "latest_checkpoint.pt"
    if not all(path.is_file() for path in (manifest_path, summary_path, checkpoint_path)):
        raise FileExistsError(
            f"refusing to reuse non-empty run directory without complete V15-C resume artifacts: {RUN_DIR}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_valid = isinstance(summary, dict)
    except json.JSONDecodeError:
        summary = {}
        summary_valid = False
    initialization = manifest.get("initialization_metadata", {})
    if (
        manifest.get("algorithm_id") != "V15-C_STATE_DEPENDENT_TRIGGER"
        or manifest.get("target_episodes") != TARGET_EPISODES
        or manifest.get("episode_seed_start") != EPISODE_SEED_START
        or manifest.get("seed") != BASE_SEED
        or not manifest.get("stage1_trigger_only")
        or initialization.get("source_checkpoint_sha256") != V14B_SHA256
        or (
            summary_valid
            and (
                summary.get("target_episodes") != TARGET_EPISODES
                or summary.get("completed_episodes")
                != manifest.get("training_counters", {}).get("completed_episodes")
            )
        )
    ):
        raise FileExistsError(
            "refusing to resume a non-empty directory with mismatched V15-C provenance."
        )
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = int(payload.get("training_counters", {}).get("completed_episodes", -1))
    normalizer = payload.get("trigger_observation_normalization", {})
    if (
        payload.get("algorithm_id") != "V15-C_STATE_DEPENDENT_TRIGGER"
        or payload.get("checkpoint_version") != 44
        or payload.get("architecture_version") != 22
        or payload.get("stage1_trigger_only") is not True
        or (
            summary_valid
            and completed != int(summary.get("completed_episodes", -2))
        )
        or not normalizer.get("active")
        or completed < 1
        or completed > TARGET_EPISODES
    ):
        raise FileExistsError("V15-C partial run checkpoint failed resume provenance checks.")
    if completed != int(manifest.get("training_counters", {}).get("completed_episodes", -1)):
        raise FileExistsError("V15-C checkpoint and manifest episode counters disagree.")
    if not summary_valid:
        normalization_payload = payload["trigger_observation_normalization"]
        episode_records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
        summary = {
            "experiment_type": "macro_cache_smdp_micro_uav_ppo",
            "algorithm_id": "V15-C_STATE_DEPENDENT_TRIGGER",
            "architecture_version": 22,
            "checkpoint_version": 44,
            "completed_episodes": completed,
            "target_episodes": TARGET_EPISODES,
            "best_episode_return": payload.get("best_episode_return"),
            "early_gate_episodes": None,
            "early_gate_status": "NOT_CONFIGURED",
            "early_stop_reason": None,
            "early_stop_detail": None,
            "training_counters": payload["training_counters"],
            "initialization_metadata": payload.get("initialization_metadata", {}),
            "stage1_trigger_only": True,
            "trigger_observation_normalization": {
                **normalization_payload,
                "mean": np.asarray(normalization_payload["mean"]).tolist(),
                "std": np.asarray(normalization_payload["std"]).tolist(),
            },
            "last_episode": episode_records[-1] if episode_records else None,
            "last_update": None,
            "resume_recovery_note": (
                "Reconstructed after summary serialization failed on NumPy normalization arrays; "
                "episode metrics and checkpoint had already been saved."
            ),
        }
        _write_json(summary_path, summary)
    episode_records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
    latest = episode_records[-1] if episode_records else {}
    latest_state_std = latest.get("trigger_p_update_state_std")
    latest_spearman = latest.get("trigger_policy_delta_spearman")
    state_dependence_failed = (
        completed >= 2
        and latest_state_std is not None
        and float(latest_state_std) < 1.5 * V15B_STATE_STD
        and (
            latest_spearman is None
            or abs(float(latest_spearman)) < 0.05
        )
    )
    if state_dependence_failed:
        EARLY_STOPPED_EXISTING_RUN = True
        summary["early_stop_reason"] = "STATE_DEPENDENT_TRIGGER_LEARNING_FAILED"
        summary["early_stop_detail"] = (
            "After episode 2, Trigger state std was below 1.5x the V15-B baseline and "
            "active-state policy/DELTA Spearman was undefined or near zero."
        )
        summary["state_dependent_trigger_learning_failed"] = True
        _write_json(summary_path, summary)
        manifest["stage1_gate_stop_reason"] = "STATE_DEPENDENT_TRIGGER_LEARNING_FAILED"
        manifest["stage1_gate_stop_episode"] = completed
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return True


def _load_trigger_normalization() -> tuple[dict[str, Any], dict[str, Any]]:
    raw_bytes = NORMALIZATION_SOURCE.read_bytes()
    source = json.loads(raw_bytes)
    states = source.get("states")
    if not isinstance(states, list) or len(states) != 200:
        raise ValueError("Trigger normalization source must contain exactly 200 states.")
    expected_seeds = set(range(20041, 20051))
    observed_seeds = {int(item["episode_seed"]) for item in states}
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Trigger normalization source seeds mismatch: {sorted(observed_seeds)}"
        )
    observations = np.asarray(
        [item["observation_400d"] for item in states], dtype=np.float64
    )
    if observations.shape != (200, 400) or not np.isfinite(observations).all():
        raise ValueError("Trigger normalization observations must be finite [200,400].")
    mean = observations.mean(axis=0, dtype=np.float64)
    raw_std = observations.std(axis=0, ddof=0, dtype=np.float64)
    floor_mask = raw_std < 1.0e-6
    std = np.maximum(raw_std, 1.0e-6)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("Trigger normalization statistics contain NaN/Inf.")
    metadata = {
        "source_path": str(NORMALIZATION_SOURCE.resolve()),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_training_episode_seeds": sorted(expected_seeds),
        "sample_count": int(len(observations)),
        "feature_dim": int(observations.shape[1]),
        "standard_deviation_ddof": 0,
        "standard_deviation_floor": 1.0e-6,
        "standard_deviation_floor_feature_count": int(floor_mask.sum()),
        "normalization_scope": "Trigger input only; other policy inputs remain raw",
    }
    normalizer = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "metadata": metadata,
    }
    return normalizer, {
        **metadata,
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def _synthetic_full_information_sign_check() -> dict[str, Any]:
    import torch

    from ppo_simple.trainer import _full_information_trigger_surrogate

    output: dict[str, Any] = {"pass": True, "clip": 0.2, "cases": {}}
    for label, delta in (("positive_delta", 1.0), ("negative_delta", -1.0)):
        logit = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optimizer = torch.optim.SGD([logit], lr=0.1)
        old_p = torch.tensor([0.5], dtype=torch.float32)
        before = float(torch.sigmoid(logit).item())
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = _full_information_trigger_surrogate(
            torch.sigmoid(logit).reshape(1),
            old_p,
            torch.tensor([delta], dtype=torch.float32),
            0.2,
        )
        (-objective.mean()).backward()
        optimizer.step()
        after = float(torch.sigmoid(logit).item())
        valid = after > before if delta > 0.0 else after < before
        if not valid:
            raise AssertionError(f"full-information sign check failed for {label}.")
        output["cases"][label] = {
            "delta": delta,
            "p_update_before": before,
            "p_update_after": after,
            "direction_pass": valid,
        }
    return output


def _optimizer_freeze_and_transplant_check(
    normalization: dict[str, Any],
) -> dict[str, Any]:
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

    def optimizer_ids(optimizer: Any) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    trigger_ids = optimizer_ids(trainer.macro_cache_trigger_actor_optimizer)
    frozen_optimizers = {
        "candidate_duration": trainer.macro_optimizer,
        "main_critic": trainer.macro_critic_optimizer,
        "long_critic": trainer.long_value_optimizer,
        "micro_actor_critic": trainer.micro_optimizer,
    }
    overlaps = {
        label: len(trigger_ids & optimizer_ids(optimizer))
        for label, optimizer in frozen_optimizers.items()
    }
    if any(overlaps.values()):
        raise AssertionError(f"Trigger optimizer overlaps frozen optimizer: {overlaps}")
    observations = torch.as_tensor(normalization["mean"], dtype=torch.float32).reshape(1, -1)
    normalized = trainer.policy._normalized_trigger_observations(observations)
    if normalized.shape != observations.shape or not bool(torch.isfinite(normalized).all()):
        raise AssertionError("Trigger normalizer finite check failed.")
    trigger_distribution = trainer.policy.cache_trigger_distribution(
        observations, torch.ones(1, dtype=torch.bool)
    )
    p_update = float(trigger_distribution.probs[0, 1].item())
    if not math.isclose(p_update, 0.5, rel_tol=0.0, abs_tol=1.0e-7):
        raise AssertionError(f"Fresh Trigger did not start at P_UPDATE=.5: {p_update}")
    non_trigger_counters = {
        name: value
        for name, value in trainer.training_counters.items()
        if name.endswith("optimizer_steps") and name != "macro_cache_trigger_optimizer_steps"
    }
    if any(non_trigger_counters.values()):
        raise AssertionError(f"Non-Trigger optimizer counters were not fresh: {non_trigger_counters}")
    return {
        "pass": True,
        "v14b_transplant": transplant,
        "normalization_finite": True,
        "trigger_initial_p_update": p_update,
        "trigger_optimizer_vs_frozen_optimizer_parameter_overlaps": overlaps,
        "fresh_non_trigger_optimizer_steps": non_trigger_counters,
        "checkpoint_rng_restored": False,
        "counterfactual_state_rng_restore_check": (
            "verified by the guarded rollout summary for each training episode"
        ),
    }


def _load_episode_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if item.get("record_type") == "episode":
                    records.append(item)
    return records


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in records if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _collapse_detected(records: list[dict[str, Any]]) -> bool:
    streak = 0
    for record in records:
        p_update = record.get("trigger_p_update_mean")
        state_std = record.get("trigger_p_update_state_std")
        if (
            p_update is not None
            and state_std is not None
            and (
                float(p_update) <= 1.0e-8
                or float(p_update) >= 1.0 - 1.0e-8
            )
            and float(state_std) <= 1.0e-6
        ):
            return True
        if p_update is None or float(p_update) < 0.02 or float(p_update) > 0.98:
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


def _summarize_run(
    train_result: dict[str, Any],
    preflight: dict[str, Any],
    normalization_metadata: dict[str, Any],
) -> dict[str, Any]:
    records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
    first10 = records[:10] if len(records) >= 10 else []
    last10 = records[-10:] if len(records) >= TARGET_EPISODES else []
    if len(records) != TARGET_EPISODES:
        episode_count_valid = False
    else:
        episode_count_valid = True
    update_checks = [
        bool(item.get("counterfactual_state_restore_valid"))
        and bool(item.get("rollout_rng_unchanged"))
        for item in records
    ]
    non_trigger_step_values = [
        int(item.get("formal_non_trigger_optimizer_steps_this_update", -1))
        for item in records
    ]
    non_trigger_optimizer_steps = int(sum(max(value, 0) for value in non_trigger_step_values))
    positive_count = sum(int(item.get("trigger_delta_positive_count", 0)) for item in records)
    negative_count = sum(int(item.get("trigger_delta_negative_count", 0)) for item in records)
    active_count = sum(int(item.get("trigger_credit_active_count", 0)) for item in records)
    positive_fraction = positive_count / max(active_count, 1)
    negative_fraction = negative_count / max(active_count, 1)
    first_return = _mean(first10, "episode_return")
    last_return = _mean(last10, "episode_return")
    return_tolerance = (
        None if first_return is None else max(25.0, 0.10 * abs(first_return))
    )
    return_not_substantially_worse = (
        first_return is not None
        and last_return is not None
        and last_return >= first_return - float(return_tolerance)
    )
    state_std_first = _mean(first10, "trigger_p_update_state_std")
    state_std_last = _mean(last10, "trigger_p_update_state_std")
    state_ratio = (
        None if state_std_last is None else state_std_last / V15B_STATE_STD
    )
    spearman_first = _mean(first10, "trigger_policy_delta_spearman")
    spearman_last = _mean(last10, "trigger_policy_delta_spearman")
    last10_spearman_values = [
        float(item["trigger_policy_delta_spearman"])
        for item in last10
        if item.get("trigger_policy_delta_spearman") is not None
    ]
    collapse = _collapse_detected(records)
    kl_mean = _mean(records, "trigger_kl_mean")
    kl_max = max(
        (float(item["trigger_kl_max"]) for item in records if item.get("trigger_kl_max") is not None),
        default=None,
    )
    epoch_early_stop_count = sum(
        int(item.get("trigger_epoch_early_stop_count", 0)) for item in records
    )
    trigger_active_fraction = _mean(records, "trigger_credit_active_fraction")
    nan_inf_free = all(_all_finite(item) for item in records)
    counterfactual_restore_pass = bool(update_checks) and all(update_checks)
    optimizer_freeze_pass = (
        len(non_trigger_step_values) == len(records)
        and all(value == 0 for value in non_trigger_step_values)
        and non_trigger_optimizer_steps == 0
    )
    latest_record = records[-1] if records else {}
    latest_state_std = latest_record.get("trigger_p_update_state_std")
    latest_spearman = latest_record.get("trigger_policy_delta_spearman")
    latest_state_ratio = (
        None if latest_state_std is None else float(latest_state_std) / V15B_STATE_STD
    )
    state_dependence_failed = (
        len(records) >= 2
        and latest_state_ratio is not None
        and latest_state_ratio < 1.5
        and (latest_spearman is None or abs(float(latest_spearman)) < 0.05)
    )
    conditions = {
        "episode_count_30": episode_count_valid,
        "trigger_not_collapsed": not collapse,
        "last10_state_std_ratio_at_least_2": (
            state_ratio is not None and state_ratio >= STATE_STD_IMPROVEMENT_RATIO_MIN
        ),
        "last10_spearman_all_positive": (
            len(last10_spearman_values) == 10
            and all(value > 0.0 for value in last10_spearman_values)
        ),
        "last10_spearman_mean_at_least_0_20": (
            spearman_last is not None and spearman_last >= SPEARMAN_LAST10_TARGET
        ),
        "both_active_delta_signs_present": positive_count > 0 and negative_count > 0,
        "return_not_substantially_worse": return_not_substantially_worse,
        "mean_trigger_kl_at_most_0_05": kl_mean is not None and kl_mean <= TRIGGER_KL_MEAN_LIMIT,
        "non_trigger_optimizers_frozen": optimizer_freeze_pass,
        "counterfactual_state_rng_restore_valid": counterfactual_restore_pass,
        "nan_inf_free": nan_inf_free,
    }
    supported = all(conditions.values())
    return {
        "experiment": "V15-C state-dependent Trigger Stage-1 convergence gate",
        "status": "COMPLETED" if episode_count_valid else "INCOMPLETE",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "episodes_completed": len(records),
        "early_stop_reason": train_result.get(
            "early_stop_reason",
            "STATE_DEPENDENT_TRIGGER_LEARNING_FAILED" if state_dependence_failed else None,
        ),
        "early_stop_detail": train_result.get("early_stop_detail"),
        "macro_actor_learning_rate": MACRO_ACTOR_LR,
        "downstream_warmstart_from_v14b": True,
        "trigger_reinitialized": True,
        "initialization_check": preflight,
        "trigger_normalization_active": True,
        "trigger_normalization_source": normalization_metadata,
        "full_information_binary_trigger_ppo": True,
        "full_information_advantage_definition": {
            "A_UPDATE": "(1-p_old)*DELTA_scaled",
            "A_KEEP": "-p_old*DELTA_scaled",
            "surrogate": "old-policy probability weighted separately clipped KEEP/UPDATE PPO objectives",
            "clip_epsilon": 0.2,
            "delta_scaling": "active-state MAD; divide by max(MAD,1e-8), clip [-5,5]",
            "critic_bootstrap": False,
        },
        "stage1_frozen_optimizers": [
            "Cache Candidate",
            "Duration",
            "Main Critic",
            "Long Critic",
            "Micro Actor/Critic",
        ],
        "formal_non_trigger_optimizer_steps": non_trigger_optimizer_steps,
        "counterfactual_state_rng_restore_pass": counterfactual_restore_pass,
        "nan_inf_free": nan_inf_free,
        "return_first10_mean": first_return,
        "return_last10_mean": last_return,
        "return_first_available_episodes_mean": _mean(records[:10], "episode_return"),
        "return_latest_episode": latest_record.get("episode_return"),
        "first10_and_last10_comparison_available": (
            len(first10) == 10 and len(last10) == 10
        ),
        "return_last10_worsening_tolerance": return_tolerance,
        "return_not_substantially_worse": return_not_substantially_worse,
        "p_update_first10": _mean(first10, "trigger_p_update_mean"),
        "p_update_last10": _mean(last10, "trigger_p_update_mean"),
        "p_update_state_std_first10": state_std_first,
        "p_update_state_std_last10": state_std_last,
        "v15b_state_std": V15B_STATE_STD,
        "state_dependence_improvement_ratio": (
            state_ratio if state_ratio is not None else latest_state_ratio
        ),
        "state_dependence_improvement_ratio_basis": (
            "last10_mean" if state_ratio is not None else "latest_episode_after_early_stop"
        ),
        "p_update_first_available_episodes": _mean(records[:10], "trigger_p_update_mean"),
        "p_update_latest_episode": latest_record.get("trigger_p_update_mean"),
        "p_update_state_std_latest_episode": latest_state_std,
        "spearman_latest_episode": latest_spearman,
        "state_std_improvement_ratio_required": STATE_STD_IMPROVEMENT_RATIO_MIN,
        "trigger_policy_delta_spearman_first10": spearman_first,
        "trigger_policy_delta_spearman_last10": spearman_last,
        "trigger_policy_delta_spearman_last10_values": last10_spearman_values,
        "trigger_policy_delta_spearman_target": SPEARMAN_LAST10_TARGET,
        "delta_positive_fraction_active": positive_fraction,
        "delta_negative_fraction_active": negative_fraction,
        "delta_positive_count_active": positive_count,
        "delta_negative_count_active": negative_count,
        "trigger_active_fraction": trigger_active_fraction,
        "trigger_entropy_mean": _mean(records, "trigger_entropy"),
        "trigger_kl_mean": kl_mean,
        "trigger_kl_max": kl_max,
        "trigger_epoch_early_stop_count": epoch_early_stop_count,
        "trigger_collapse": collapse,
        "state_dependent_trigger_learning_failed": state_dependence_failed,
        "state_dependent_trigger_learning_supported": supported,
        "ready_for_stage2": supported,
        "stage2_started": False,
        "support_conditions": conditions,
        "per_episode_metrics": records,
        "run_training_result": train_result,
    }


def main() -> int:
    resume = _resume_own_partial_run_if_valid()
    required_paths = [CONFIG_PATH, NORMALIZATION_SOURCE, V14B_CHECKPOINT]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required V15-C input is missing: {missing}")
    expected_realizations = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing_realizations = [str(path) for path in expected_realizations if not path.is_dir()]
    if missing_realizations:
        raise FileNotFoundError(
            "V15-C requires the existing 30 training realizations; generation is disabled: "
            f"{missing_realizations}"
        )

    normalization, normalization_metadata = _load_trigger_normalization()
    synthetic_check = _synthetic_full_information_sign_check()
    freeze_check = _optimizer_freeze_and_transplant_check(normalization)
    preflight = {
        "synthetic_full_information_sign_check": synthetic_check,
        "checkpoint_transplant_normalization_optimizer_check": freeze_check,
    }
    _write_json(
        REPORT_PATH,
        {
            "experiment": "V15-C state-dependent Trigger Stage-1 convergence gate",
            "status": "PREFLIGHT_PASSED",
            "formal_runner_path": str(Path(__file__).resolve()),
            "formal_command": FORMAL_COMMAND,
            "preflight": preflight,
            "trigger_normalization_source": normalization_metadata,
            "stage2_started": False,
        },
    )

    if EARLY_STOPPED_EXISTING_RUN:
        summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
        train_result = {
            "run_dir": str(RUN_DIR.resolve()),
            "completed_episodes": int(summary["completed_episodes"]),
            "target_episodes": TARGET_EPISODES,
            "checkpoint": str((RUN_DIR / "checkpoints" / "latest_checkpoint.pt").resolve()),
            "best_episode_return": summary.get("best_episode_return"),
            "early_stop_reason": summary.get("early_stop_reason"),
            "early_stop_detail": summary.get("early_stop_detail"),
        }
        result = _summarize_run(train_result, preflight, normalization_metadata)
        result["status"] = "STOPPED_EARLY_FAILURE"
        _write_json(REPORT_PATH, result)
        _write_json(RUN_DIR / "v15c_stage1_gate_report.json", result)
        return 0

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
        resume=resume,
        early_gate_episodes=None,
        initial_checkpoint_path=None if resume else V14B_CHECKPOINT,
        expected_initial_checkpoint_sha256=V14B_SHA256,
        stage1_trigger_only=True,
        trigger_observation_normalization=normalization,
    )
    result = _summarize_run(train_result, preflight, normalization_metadata)
    _write_json(REPORT_PATH, result)
    _write_json(RUN_DIR / "v15c_stage1_gate_report.json", result)
    print(
        json.dumps(
            {
                "episodes_completed": result["episodes_completed"],
                "return_first10_mean": result["return_first10_mean"],
                "return_last10_mean": result["return_last10_mean"],
                "p_update_first10": result["p_update_first10"],
                "p_update_last10": result["p_update_last10"],
                "p_update_state_std_last10": result["p_update_state_std_last10"],
                "state_dependence_improvement_ratio": result[
                    "state_dependence_improvement_ratio"
                ],
                "trigger_policy_delta_spearman_last10": result[
                    "trigger_policy_delta_spearman_last10"
                ],
                "trigger_epoch_early_stop_count": result[
                    "trigger_epoch_early_stop_count"
                ],
                "state_dependent_trigger_learning_supported": result[
                    "state_dependent_trigger_learning_supported"
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
