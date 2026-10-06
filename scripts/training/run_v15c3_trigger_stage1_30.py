from __future__ import annotations

import copy
import hashlib
import importlib.util
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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15c3_trigger_stage1_30"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15c3_trigger_stage1_30.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
C2_RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_v15c2_trigger_normalization_gate.py"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_EPISODES = 30
MACRO_ACTOR_LR = 1.0e-3
TARGET_KL = 0.02
HARD_KL = 0.05
BACKTRACKING_SCALES = [1.0, 0.25, 0.0625, 0.015625]
MAX_GRAD_NORM = 0.5
LOW_VARIANCE_FEATURE_COUNT = 73
TRIGGER_INPUT_CLIP = [-10.0, 10.0]
EXPECTED_NORMALIZATION_SOURCE_SHA256 = (
    "1be40fb05ffce97bf3fe58085c1229661feb8b5c92f4556010b4ab5396324af2"
)
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15c3_trigger_stage1_30.py"
)


def _load_c2_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("v15c2_normalization_gate", C2_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the verified V15-C2 runner: {C2_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _assert_fresh_run_dir() -> None:
    resolved = RUN_DIR.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if resolved.parent != runs_root:
        raise ValueError(f"C3 run directory must be a direct child of runs/: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty C3 run directory: {resolved}")


def _load_episode_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    if item.get("record_type") == "episode":
                        records.append(item)
    return records


def _resume_state() -> tuple[bool, int, str | None]:
    if not RUN_DIR.exists():
        return False, 0, None
    if not RUN_DIR.is_dir() or not any(RUN_DIR.iterdir()):
        return False, 0, None
    if not REPORT_PATH.is_file():
        raise FileExistsError("refusing to resume C3 artifacts without its provenance report")
    prior_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if (
        prior_report.get("experiment") != "V15-C3 full Stage-1 Trigger convergence gate"
        or prior_report.get("normalization_source", {}).get("source_sha256")
        != EXPECTED_NORMALIZATION_SOURCE_SHA256
    ):
        raise FileExistsError("existing C3 run/report provenance does not match this protocol")

    manifest_path = RUN_DIR / "run_manifest.json"
    summary_path = RUN_DIR / "run_summary.json"
    checkpoint_path = RUN_DIR / "checkpoints" / "latest_checkpoint.pt"
    metrics_path = RUN_DIR / "episode_metrics.jsonl"
    required = (manifest_path, summary_path, checkpoint_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileExistsError(f"C3 run has no verifiable checkpoint prefix: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "algorithm_id": "V15-C2_TRIGGER_NORMALIZATION",
        "architecture_version": 22,
        "checkpoint_version": 46,
        "target_episodes": TARGET_EPISODES,
        "episode_seed_start": EPISODE_SEED_START,
        "seed": BASE_SEED,
        "stage1_trigger_only": True,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise FileExistsError("existing run manifest does not match the C3 C2-based protocol")
    norm_metadata = manifest.get("trigger_observation_normalization_metadata", {})
    if (
        norm_metadata.get("source_sha256") != EXPECTED_NORMALIZATION_SOURCE_SHA256
        or norm_metadata.get("scale_rule") != "unit_scale_for_low_variance"
        or norm_metadata.get("input_clip") != TRIGGER_INPUT_CLIP
    ):
        raise FileExistsError("existing run manifest has a mismatched Trigger normalizer")
    if any(
        summary.get(key) != value
        for key, value in (
            ("algorithm_id", expected_manifest["algorithm_id"]),
            ("architecture_version", expected_manifest["architecture_version"]),
            ("checkpoint_version", expected_manifest["checkpoint_version"]),
            ("target_episodes", TARGET_EPISODES),
            ("stage1_trigger_only", True),
        )
    ):
        raise FileExistsError("existing run summary does not match the C3 protocol")
    records = _load_episode_metrics(metrics_path)
    episode_numbers = [int(record.get("episode", -1)) for record in records]
    if episode_numbers != list(range(1, len(episode_numbers) + 1)) or len(records) > TARGET_EPISODES:
        raise FileExistsError(f"C3 episode records are not a valid prefix: {episode_numbers}")
    expected_seeds = list(range(EPISODE_SEED_START, EPISODE_SEED_START + len(records)))
    actual_seeds = [int(record.get("episode_seed", -1)) for record in records]
    if actual_seeds != expected_seeds:
        raise FileExistsError(f"C3 episode seed prefix mismatch: {actual_seeds}")
    completed = int(summary.get("completed_episodes", -1))
    if completed != len(records):
        raise FileExistsError("C3 summary and episode metrics disagree on completed episodes")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        int(checkpoint.get("checkpoint_version", -1)) != 46
        or int(checkpoint.get("architecture_version", -1)) != 22
        or int(checkpoint.get("training_counters", {}).get("completed_episodes", -1)) != completed
        or checkpoint.get("stage1_trigger_only") is not True
    ):
        raise FileExistsError("C3 latest checkpoint metadata disagrees with its prefix")
    stop_reason = summary.get("early_stop_reason")
    if completed == TARGET_EPISODES or stop_reason:
        return False, completed, str(stop_reason) if stop_reason else None
    return True, completed, None


def _nested_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _transaction_rollback_check() -> dict[str, Any]:
    import torch

    from ppo_simple.trainer import _transactional_optimizer_update

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=MACRO_ACTOR_LR)
    warmup_loss = model(torch.ones(1, 3)).square().sum()
    warmup_loss.backward()
    optimizer.step()
    before_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())

    def step_fn() -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        loss = -model(torch.full((1, 3), 100.0)).sum()
        loss.backward()
        preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM).item())
        return {"pre_clip_grad_norm": preclip}

    outcome = _transactional_optimizer_update(
        model,
        optimizer,
        step_fn=step_fn,
        kl_fn=lambda: (HARD_KL + 1.0e-6, 0.2),
        hard_kl=HARD_KL,
        step_scales=tuple(BACKTRACKING_SCALES),
    )
    model_restored = all(
        torch.equal(model.state_dict()[key], value) for key, value in before_model.items()
    )
    optimizer_restored = _nested_equal(optimizer.state_dict(), before_optimizer)
    if outcome["accepted"] or outcome["rollback_count"] != len(BACKTRACKING_SCALES):
        raise AssertionError("C3 transactional KL retry/reject protocol did not reject all retries.")
    if not model_restored or not optimizer_restored:
        raise AssertionError("C3 transactional rollback did not restore model and optimizer exactly.")
    return {
        "pass": True,
        "forced_mean_kl": HARD_KL + 1.0e-6,
        "backtracking_scales": list(BACKTRACKING_SCALES),
        "attempt_count": int(outcome["attempt_count"]),
        "rollback_count": int(outcome["rollback_count"]),
        "model_bitwise_restored": model_restored,
        "optimizer_bitwise_restored": optimizer_restored,
    }


def _validate_inputs_and_preflight(
    c2: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_files = [CONFIG_PATH, V14B_CHECKPOINT, c2.V15C_NORMALIZATION_CHECKPOINT, c2.NORMALIZATION_SOURCE]
    missing = [str(path) for path in input_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"C3 inputs missing: {missing}")
    if hashlib.sha256(V14B_CHECKPOINT.read_bytes()).hexdigest() != V14B_SHA256:
        raise ValueError("V14b episode-300 warm-start checkpoint SHA256 mismatch.")
    realization_paths = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing_realizations = [str(path) for path in realization_paths if not path.is_dir()]
    if missing_realizations:
        raise FileNotFoundError(f"C3 requires existing realizations: {missing_realizations}")

    normalization, source = c2._load_training_only_normalizer()
    if source.get("source_sha256") != EXPECTED_NORMALIZATION_SOURCE_SHA256:
        raise ValueError("C3 normalization source SHA256 mismatch.")
    if int(source.get("low_variance_feature_count", -1)) != LOW_VARIANCE_FEATURE_COUNT:
        raise ValueError("C3 normalization low-variance feature count is not 73.")
    raw_std = np.asarray(normalization["raw_std"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    expected_scale = np.where(raw_std <= 1.0e-6, 1.0, raw_std)
    if not np.array_equal(scale, expected_scale):
        raise AssertionError("C3 normalization no longer follows the verified V15-C2 scale rule.")
    preflight = c2._preflight(normalization)
    transaction_check = _transaction_rollback_check()
    preflight["transactional_kl_rollback_check"] = transaction_check
    preflight["seed_realization_pool_complete"] = len(realization_paths) == TARGET_EPISODES
    preflight["seed_realization_directories"] = [str(path.resolve()) for path in realization_paths]
    preflight["normalization_postclip_limit"] = max(abs(value) for value in TRIGGER_INPUT_CLIP)
    preflight["normalization_low_variance_feature_count"] = LOW_VARIANCE_FEATURE_COUNT
    preflight["teacher_recomputed"] = False
    return normalization, source, preflight


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _episode_projection(record: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "episode": "episode",
        "episode_seed": "episode_seed",
        "return": "episode_return",
        "p_update_mean": "trigger_p_update_after",
        "p_update_state_std": "trigger_p_update_state_std",
        "actual_cache_update_rate": "actual_cache_update_rate",
        "trigger_entropy": "trigger_entropy",
        "policy_delta_spearman": "trigger_policy_delta_spearman",
        "trigger_active_fraction": "trigger_credit_active_fraction",
        "delta_positive_fraction": "trigger_delta_positive_fraction",
        "delta_negative_fraction": "trigger_delta_negative_fraction",
        "trigger_kl_mean": "trigger_kl_mean",
        "trigger_kl_max": "trigger_kl_max",
        "accepted_epoch_count": "trigger_epoch_accepted_count",
        "rejected_epoch_count": "trigger_epoch_rejected_count",
        "rollback_count": "trigger_rollback_count_this_update",
        "preclip_grad_norm": "trigger_preclip_grad_norm",
        "postclip_grad_norm": "trigger_postclip_grad_norm",
        "normalized_observation_preclip_abs_max": "trigger_preclip_normalized_obs_abs_max",
        "normalized_observation_postclip_abs_max": "trigger_postclip_normalized_obs_abs_max",
        "normalized_observation_clipped_value_fraction": "trigger_normalized_obs_clipped_value_fraction",
        "non_trigger_optimizer_steps": "formal_non_trigger_optimizer_steps_this_update",
        "trigger_optimizer_steps": "macro_cache_trigger_optimizer_steps_this_update",
    }
    projection = {name: record.get(source) for name, source in fields.items()}
    projection["trigger_epoch_records"] = record.get("trigger_epoch_records", [])
    projection["actual_return"] = record.get("episode_return")
    projection["cache_action_index_summary"] = record.get("cache_action_pattern")
    projection["requested_duration_mean"] = record.get("requested_duration_mean")
    projection["positive_delta_active_samples"] = record.get("trigger_delta_positive_count")
    projection["negative_delta_active_samples"] = record.get("trigger_delta_negative_count")
    return projection


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _accepted_attempts(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        for epoch in record.get("trigger_epoch_records", []):
            for attempt in epoch.get("attempts", []):
                item = {
                    "episode": int(record.get("episode", -1)),
                    "epoch": int(epoch.get("epoch", -1)),
                    "attempt": int(attempt.get("attempt", -1)),
                    "step_scale": float(attempt.get("step_scale", 0.0)),
                    "kl_mean": float(attempt.get("kl_mean", 0.0)),
                    "kl_max": float(attempt.get("kl_max", 0.0)),
                    "accepted": bool(attempt.get("accepted")),
                }
                (accepted if item["accepted"] else rejected).append(item)
    return accepted, rejected


def _build_report(
    train_result: dict[str, Any],
    preflight: dict[str, Any],
    normalization_source: dict[str, Any],
) -> dict[str, Any]:
    records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
    projected = [_episode_projection(record) for record in records]
    accepted, rejected = _accepted_attempts(records)
    accepted_kls = [item["kl_mean"] for item in accepted]
    accepted_state_kls = [item["kl_max"] for item in accepted]
    rejected_epoch_count = sum(
        int(record.get("trigger_epoch_rejected_count", 0)) for record in records
    )
    accepted_epoch_count = sum(
        int(record.get("trigger_epoch_accepted_count", 0)) for record in records
    )
    rollback_count = sum(int(record.get("trigger_rollback_count_this_update", 0)) for record in records)
    groups: dict[str, list[dict[str, Any]]] = {
        "episodes_1_10": records[:10],
        "episodes_11_20": records[10:20],
        "episodes_21_30": records[20:30],
    }
    summary_field_map = {
        "RETURN_MEAN": "episode_return",
        "P_UPDATE_MEAN": "trigger_p_update_after",
        "P_UPDATE_STATE_STD_MEAN": "trigger_p_update_state_std",
        "POLICY_DELTA_SPEARMAN_MEAN": "trigger_policy_delta_spearman",
        "TRIGGER_ENTROPY_MEAN": "trigger_entropy",
        "CACHE_UPDATE_RATE_MEAN": "actual_cache_update_rate",
        "ACTIVE_FRACTION_MEAN": "trigger_credit_active_fraction",
        "KL_MEAN": "trigger_kl_mean",
    }
    stage_summaries = {
        group_name: {output_name: _mean(group, input_name) for output_name, input_name in summary_field_map.items()}
        for group_name, group in groups.items()
    }
    first10 = groups["episodes_1_10"]
    middle10 = groups["episodes_11_20"]
    last10 = groups["episodes_21_30"]
    p_last = _mean(last10, "trigger_p_update_after")
    std_last = _mean(last10, "trigger_p_update_state_std")
    spearman_last = _mean(last10, "trigger_policy_delta_spearman")
    last_pos = sum(float(record.get("trigger_delta_positive_fraction", 0.0) or 0.0) > 0.0 for record in last10)
    last_neg = sum(float(record.get("trigger_delta_negative_fraction", 0.0) or 0.0) > 0.0 for record in last10)
    collapse = bool(
        (p_last is not None and not (0.05 < p_last < 0.95))
        or train_result.get("early_stop_reason") in {
            "TRIGGER_COLLAPSE_LOW_P_UPDATE",
            "TRIGGER_COLLAPSE_HIGH_P_UPDATE",
        }
    )
    normalization_maxima = [
        float(record["trigger_postclip_normalized_obs_abs_max"])
        for record in records
        if record.get("trigger_postclip_normalized_obs_abs_max") is not None
    ]
    normalization_stable = bool(
        preflight.get("pass")
        and preflight.get("normalization_unit_and_clip_check", {}).get("pass")
        and len(normalization_maxima) == len(records)
        and all(value <= 10.0 for value in normalization_maxima)
    )
    finite_fields = (
        "episode_return",
        "trigger_p_update_after",
        "trigger_p_update_state_std",
        "actual_cache_update_rate",
        "trigger_entropy",
        "trigger_policy_delta_spearman",
        "trigger_credit_active_fraction",
        "trigger_delta_positive_fraction",
        "trigger_delta_negative_fraction",
        "trigger_kl_mean",
        "trigger_kl_max",
        "trigger_preclip_grad_norm",
        "trigger_postclip_grad_norm",
        "trigger_preclip_normalized_obs_abs_max",
        "trigger_postclip_normalized_obs_abs_max",
        "trigger_normalized_obs_clipped_value_fraction",
    )
    finite_records = all(
        _finite({key: record.get(key) for key in finite_fields})
        for record in records
    ) and all(_finite(item) for item in accepted + rejected)
    accepted_kl_safe = all(value <= HARD_KL for value in accepted_kls)
    non_trigger_steps = 0
    run_summary_path = RUN_DIR / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.is_file() else {}
    non_trigger_steps = int(
        run_summary.get("training_counters", {}).get("formal_non_trigger_optimizer_steps", 0)
    )
    optimizer_freeze_ok = (
        non_trigger_steps == 0
        and all(int(record.get("formal_non_trigger_optimizer_steps_this_update", -1)) == 0 for record in records)
    )
    early_stop_reason = train_result.get("early_stop_reason") or run_summary.get("early_stop_reason")
    completed = len(records)
    implementation_complete = bool(
        completed == TARGET_EPISODES
        and int(train_result.get("completed_episodes", completed)) == TARGET_EPISODES
        and preflight.get("pass")
        and finite_records
        and optimizer_freeze_ok
        and normalization_stable
        and accepted_kl_safe
    )
    state_supported = bool(
        implementation_complete
        and not collapse
        and p_last is not None
        and 0.05 < p_last < 0.95
        and std_last is not None
        and std_last > 0.02
        and spearman_last is not None
        and spearman_last >= 0.20
        and last_pos > 0
        and last_neg > 0
        and accepted_kl_safe
        and normalization_stable
    )
    if state_supported:
        failure_mode = "NONE"
    elif collapse:
        failure_mode = "TRIGGER_COLLAPSE"
    elif not accepted_kl_safe or early_stop_reason == "ACCEPTED_KL_ABOVE_HARD_LIMIT":
        failure_mode = "KL_INSTABILITY"
    elif std_last is not None and std_last <= 0.02:
        failure_mode = "STATE_DEPENDENCE_LOST"
    elif spearman_last is not None and (spearman_last < 0.20 or last_pos == 0 or last_neg == 0):
        failure_mode = "TEACHER_ALIGNMENT_LOST"
    elif implementation_complete and not state_supported:
        failure_mode = "RETURN_DEGRADATION_WITH_GOOD_ALIGNMENT"
    elif early_stop_reason in {"NAN_INF", "NORMALIZATION_POSTCLIP_ABOVE_10"}:
        failure_mode = "KL_INSTABILITY" if early_stop_reason == "NAN_INF" else "STATE_DEPENDENCE_LOST"
    else:
        failure_mode = "INCOMPLETE_RUN"

    return {
        "experiment": "V15-C3 full Stage-1 Trigger convergence gate",
        "status": (
            "COMPLETED"
            if completed == TARGET_EPISODES and not early_stop_reason
            else "EARLY_STOPPED"
            if early_stop_reason
            else "IN_PROGRESS"
        ),
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "algorithm_mechanism": "V15-C2 Trigger Stage-1, unchanged",
        "architecture_version": 22,
        "checkpoint_version": 46,
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "episodes_completed": completed,
        "downstream_warmstart_from_v14b_episode_300": True,
        "v14b_checkpoint_sha256": V14B_SHA256,
        "trigger_fresh_reinitialization": True,
        "fresh_trigger_initial_p_update": preflight.get("fresh_trigger_initial_p_update"),
        "stage1_trigger_only": True,
        "trigger_learning_rate": MACRO_ACTOR_LR,
        "trigger_target_kl": TARGET_KL,
        "trigger_hard_kl": HARD_KL,
        "trigger_backtracking_step_scales": BACKTRACKING_SCALES,
        "trigger_max_grad_norm": MAX_GRAD_NORM,
        "normalization_source": normalization_source,
        "normalization_protocol": {
            "raw_std_threshold": 1.0e-6,
            "low_variance_scale": 1.0,
            "other_feature_scale": "raw_std",
            "input_clip": TRIGGER_INPUT_CLIP,
            "low_variance_feature_count": LOW_VARIANCE_FEATURE_COUNT,
            "scope": "Trigger only; Candidate/Duration/Critic/Micro inputs unchanged",
            "running_normalization": False,
        },
        "early_stop_protocol": {
            "nan_inf": True,
            "p_update_below_0_02_or_above_0_98_consecutive_episodes": 3,
            "p_update_state_std_below_0_002_consecutive_episodes": 3,
            "postclip_abs_max_above_10": True,
            "accepted_mean_kl_above_0_05": True,
        },
        "preflight": preflight,
        "stage_summaries": stage_summaries,
        "episodes": projected,
        "accepted_kl_mean_values": accepted_kls,
        "accepted_kl_per_state_max_values": accepted_state_kls,
        "rejected_attempt_kl_mean_values": [item["kl_mean"] for item in rejected],
        "accepted_kl_max": max(accepted_kls) if accepted_kls else None,
        "accepted_kl_mean": float(np.mean(accepted_kls)) if accepted_kls else None,
        "accepted_kl_per_state_max": max(accepted_state_kls) if accepted_state_kls else None,
        "accepted_epoch_count": accepted_epoch_count,
        "rejected_epoch_count": rejected_epoch_count,
        "rejected_attempt_count": len(rejected),
        "rollback_count": rollback_count,
        "non_trigger_optimizer_steps": non_trigger_steps,
        "normalization_stable": normalization_stable,
        "nan_inf_free": finite_records,
        "optimizer_freeze_verified": optimizer_freeze_ok,
        "early_stop_reason": early_stop_reason,
        "run_training_result": train_result,
        "IMPLEMENTATION_COMPLETE": "YES" if implementation_complete else "NO",
        "EPISODES_COMPLETED": completed,
        "NON_TRIGGER_OPTIMIZER_STEPS": non_trigger_steps,
        "RETURN_FIRST10_MEAN": stage_summaries["episodes_1_10"]["RETURN_MEAN"],
        "RETURN_MIDDLE10_MEAN": stage_summaries["episodes_11_20"]["RETURN_MEAN"],
        "RETURN_LAST10_MEAN": stage_summaries["episodes_21_30"]["RETURN_MEAN"],
        "P_UPDATE_FIRST10_MEAN": stage_summaries["episodes_1_10"]["P_UPDATE_MEAN"],
        "P_UPDATE_MIDDLE10_MEAN": stage_summaries["episodes_11_20"]["P_UPDATE_MEAN"],
        "P_UPDATE_LAST10_MEAN": stage_summaries["episodes_21_30"]["P_UPDATE_MEAN"],
        "P_UPDATE_STATE_STD_FIRST10_MEAN": stage_summaries["episodes_1_10"]["P_UPDATE_STATE_STD_MEAN"],
        "P_UPDATE_STATE_STD_MIDDLE10_MEAN": stage_summaries["episodes_11_20"]["P_UPDATE_STATE_STD_MEAN"],
        "P_UPDATE_STATE_STD_LAST10_MEAN": stage_summaries["episodes_21_30"]["P_UPDATE_STATE_STD_MEAN"],
        "POLICY_DELTA_SPEARMAN_FIRST10_MEAN": stage_summaries["episodes_1_10"]["POLICY_DELTA_SPEARMAN_MEAN"],
        "POLICY_DELTA_SPEARMAN_MIDDLE10_MEAN": stage_summaries["episodes_11_20"]["POLICY_DELTA_SPEARMAN_MEAN"],
        "POLICY_DELTA_SPEARMAN_LAST10_MEAN": stage_summaries["episodes_21_30"]["POLICY_DELTA_SPEARMAN_MEAN"],
        "TRIGGER_ENTROPY_FIRST10_MEAN": stage_summaries["episodes_1_10"]["TRIGGER_ENTROPY_MEAN"],
        "TRIGGER_ENTROPY_LAST10_MEAN": stage_summaries["episodes_21_30"]["TRIGGER_ENTROPY_MEAN"],
        "CACHE_UPDATE_RATE_LAST10_MEAN": stage_summaries["episodes_21_30"]["CACHE_UPDATE_RATE_MEAN"],
        "TRIGGER_ACTIVE_FRACTION_MEAN": _mean(records, "trigger_credit_active_fraction"),
        "ACCEPTED_KL_MAX": max(accepted_kls) if accepted_kls else None,
        "ACCEPTED_KL_MEAN": float(np.mean(accepted_kls)) if accepted_kls else None,
        "TOTAL_ACCEPTED_EPOCHS": accepted_epoch_count,
        "TOTAL_REJECTED_EPOCHS": rejected_epoch_count,
        "TOTAL_ROLLBACKS": rollback_count,
        "NORMALIZATION_STABLE": "YES" if normalization_stable else "NO",
        "TRIGGER_COLLAPSE": "YES" if collapse else "NO",
        "STATE_DEPENDENT_TRIGGER_LEARNING_SUPPORTED": "YES" if state_supported else "NO",
        "READY_FOR_STAGE2": "YES" if state_supported else "NO",
        "PRIMARY_FAILURE_MODE": failure_mode,
    }


def _episode_stop_callback(
    preflight: dict[str, Any], normalization_source: dict[str, Any]
) -> Any:
    def callback(episode_record: dict[str, Any], update_record: dict[str, Any]) -> str | None:
        all_records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
        p_low_streak = 0
        p_high_streak = 0
        std_low_streak = 0
        stop_reason: str | None = None
        if not _finite(episode_record) or not _finite(update_record):
            stop_reason = "NAN_INF"
        for record in all_records:
            p_value = record.get("trigger_p_update_after")
            std_value = record.get("trigger_p_update_state_std")
            p_low_streak = p_low_streak + 1 if p_value is not None and float(p_value) < 0.02 else 0
            p_high_streak = p_high_streak + 1 if p_value is not None and float(p_value) > 0.98 else 0
            std_low_streak = std_low_streak + 1 if std_value is not None and float(std_value) < 0.002 else 0
        postclip_max = episode_record.get("trigger_postclip_normalized_obs_abs_max")
        if stop_reason is None and postclip_max is not None and float(postclip_max) > 10.0:
            stop_reason = "NORMALIZATION_POSTCLIP_ABOVE_10"
        accepted_means = [
            float(attempt["kl_mean"])
            for epoch in episode_record.get("trigger_epoch_records", [])
            for attempt in epoch.get("attempts", [])
            if attempt.get("accepted")
        ]
        if stop_reason is None and any(value > HARD_KL for value in accepted_means):
            stop_reason = "ACCEPTED_KL_ABOVE_HARD_LIMIT"
        if stop_reason is None and p_low_streak >= 3:
            stop_reason = "TRIGGER_COLLAPSE_LOW_P_UPDATE"
        if stop_reason is None and p_high_streak >= 3:
            stop_reason = "TRIGGER_COLLAPSE_HIGH_P_UPDATE"
        if stop_reason is None and std_low_streak >= 3:
            stop_reason = "TRIGGER_STATE_STD_BELOW_0_002_FOR_3_EPISODES"

        current_result = {
            "run_dir": str(RUN_DIR.resolve()),
            "completed_episodes": len(all_records),
            "target_episodes": TARGET_EPISODES,
            "early_stop_reason": stop_reason,
        }
        partial = _build_report(current_result, preflight, normalization_source)
        if stop_reason is None and len(all_records) < TARGET_EPISODES:
            partial["status"] = "IN_PROGRESS"
        _write_json(REPORT_PATH, partial)
        return stop_reason

    return callback


def main() -> int:
    c2 = _load_c2_helpers()
    resume_existing, existing_completed, prior_stop_reason = _resume_state()
    if not resume_existing and existing_completed == 0:
        _assert_fresh_run_dir()
    normalization, normalization_source, preflight = _validate_inputs_and_preflight(c2)
    _write_json(
        REPORT_PATH,
        {
            "experiment": "V15-C3 full Stage-1 Trigger convergence gate",
            "status": "PREFLIGHT_PASSED",
            "formal_runner_path": str(Path(__file__).resolve()),
            "formal_command": FORMAL_COMMAND,
            "run_dir": str(RUN_DIR.resolve()),
            "report_path": str(REPORT_PATH.resolve()),
            "base_seed": BASE_SEED,
            "episode_seed_start": EPISODE_SEED_START,
            "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
            "target_episodes": TARGET_EPISODES,
            "normalization_source": normalization_source,
            "preflight": preflight,
        },
    )

    records = _load_episode_metrics(RUN_DIR / "episode_metrics.jsonl")
    if prior_stop_reason or existing_completed == TARGET_EPISODES:
        run_summary = json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
        train_result = {
            "run_dir": str(RUN_DIR.resolve()),
            "completed_episodes": existing_completed,
            "target_episodes": TARGET_EPISODES,
            "checkpoint": str((RUN_DIR / "checkpoints" / "latest_checkpoint.pt").resolve()),
            "early_stop_reason": prior_stop_reason,
            "early_gate_passed": False,
            "best_episode_return": run_summary.get("best_episode_return"),
        }
    else:
        from ppo_simple.trainer import run_training

        if resume_existing and existing_completed < 0:
            raise RuntimeError("Invalid C3 resume prefix.")
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
            expected_initial_checkpoint_sha256=None if resume_existing else V14B_SHA256,
            stage1_trigger_only=True,
            trigger_observation_normalization=normalization,
            episode_stop_callback=_episode_stop_callback(preflight, normalization_source),
        )

    result = _build_report(train_result, preflight, normalization_source)
    _write_json(REPORT_PATH, result)
    _write_json(RUN_DIR / "v15c3_trigger_stage1_30_report.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "IMPLEMENTATION_COMPLETE",
                    "EPISODES_COMPLETED",
                    "NON_TRIGGER_OPTIMIZER_STEPS",
                    "RETURN_FIRST10_MEAN",
                    "RETURN_MIDDLE10_MEAN",
                    "RETURN_LAST10_MEAN",
                    "P_UPDATE_FIRST10_MEAN",
                    "P_UPDATE_MIDDLE10_MEAN",
                    "P_UPDATE_LAST10_MEAN",
                    "P_UPDATE_STATE_STD_LAST10_MEAN",
                    "POLICY_DELTA_SPEARMAN_LAST10_MEAN",
                    "ACCEPTED_KL_MAX",
                    "TOTAL_ACCEPTED_EPOCHS",
                    "TOTAL_REJECTED_EPOCHS",
                    "TOTAL_ROLLBACKS",
                    "NORMALIZATION_STABLE",
                    "TRIGGER_COLLAPSE",
                    "STATE_DEPENDENT_TRIGGER_LEARNING_SUPPORTED",
                    "READY_FOR_STAGE2",
                    "PRIMARY_FAILURE_MODE",
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
