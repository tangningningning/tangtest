from __future__ import annotations

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
RUN_DIR = PROJECT_ROOT / "runs" / "formal_v15c4_trigger_sync_r4_10"
REPORT_PATH = PROJECT_ROOT / "reports" / "v15c4_trigger_sync_r4_10.json"
V14B_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "formal_v14b_actor_lr_1e_minus_3"
    / "checkpoints"
    / "latest_checkpoint.pt"
)
V14B_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"
C2_RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_v15c2_trigger_normalization_gate.py"
C3_RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_v15c3_trigger_stage1_30.py"
BASE_SEED = 27005
EPISODE_SEED_START = 20001
TARGET_UPDATES = 10
REALIZATIONS_PER_UPDATE = 4
TARGET_EPISODES = TARGET_UPDATES * REALIZATIONS_PER_UPDATE
MACRO_ACTOR_LR = 1.0e-3
HARD_KL = 0.05
TRIGGER_INPUT_CLIP = [-10.0, 10.0]
EXPECTED_NORMALIZATION_SOURCE_SHA256 = (
    "1be40fb05ffce97bf3fe58085c1229661feb8b5c92f4556010b4ab5396324af2"
)
FORMAL_COMMAND = (
    "conda activate adaptive_cache_mec; "
    "python scripts/training/run_v15c4_trigger_sync_r4_10.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _read_records(path: Path, record_type: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if record.get("record_type") == record_type:
                        records.append(record)
    return records


def _equal_weight_loss_check() -> dict[str, Any]:
    import torch

    from ppo_simple.trainer import _equal_realization_tensor_mean

    values = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)
    groups = np.asarray([0, 0, 1], dtype=np.int64)
    loss = _equal_realization_tensor_mean(values, groups)
    loss.backward()
    expected_loss = 4.5
    expected_gradient = np.asarray([0.25, 0.25, 0.5], dtype=np.float64)
    if not math.isclose(float(loss.detach()), expected_loss, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError("Synthetic synchronized loss did not give each realization equal weight.")
    if not np.allclose(values.grad.detach().numpy(), expected_gradient, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Synthetic synchronized loss gradients are not realization-equal.")
    return {
        "pass": True,
        "realization_losses": [2.0, 7.0],
        "equal_weight_loss": float(loss.detach()),
        "pooled_sample_weighted_loss": float(np.mean([1.0, 3.0, 7.0])),
        "sample_gradients": values.grad.detach().numpy().tolist(),
    }


def _validate_inputs_and_preflight(c2: Any, c3: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = [CONFIG_PATH, V14B_CHECKPOINT, c2.V15C_NORMALIZATION_CHECKPOINT, c2.NORMALIZATION_SOURCE]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"C4 inputs missing: {missing}")
    actual_v14b_sha = hashlib.sha256(V14B_CHECKPOINT.read_bytes()).hexdigest()
    if actual_v14b_sha != V14B_SHA256:
        raise ValueError("V14b episode-300 warm-start checkpoint SHA256 mismatch.")
    realization_paths = [
        EPISODE_POOL / f"episode_{seed}"
        for seed in range(EPISODE_SEED_START, EPISODE_SEED_START + TARGET_EPISODES)
    ]
    missing_realizations = [str(path) for path in realization_paths if not path.is_dir()]
    if missing_realizations:
        raise FileNotFoundError(f"C4 requires existing realizations: {missing_realizations}")

    c3_report_path = PROJECT_ROOT / "reports" / "v15c3_trigger_stage1_30.json"
    if not c3_report_path.is_file():
        raise FileNotFoundError("Verified V15-C3 report is missing.")
    c3_report = json.loads(c3_report_path.read_text(encoding="utf-8"))
    if (
        c3_report.get("experiment") != "V15-C3 full Stage-1 Trigger convergence gate"
        or c3_report.get("normalization_source", {}).get("source_sha256")
        != EXPECTED_NORMALIZATION_SOURCE_SHA256
        or c3_report.get("EPISODES_COMPLETED") != 30
    ):
        raise ValueError("V15-C3 baseline report does not match the verified 30-episode protocol.")

    normalization, source = c2._load_training_only_normalizer()
    if source.get("source_sha256") != EXPECTED_NORMALIZATION_SOURCE_SHA256:
        raise ValueError("C4 normalization source SHA256 mismatch.")
    raw_std = np.asarray(normalization["raw_std"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    if not np.array_equal(scale, np.where(raw_std <= 1.0e-6, 1.0, raw_std)):
        raise AssertionError("C4 Trigger normalization does not follow the V15-C2 scale rule.")
    preflight = c2._preflight(normalization)
    preflight["transactional_kl_rollback_check"] = c3._transaction_rollback_check()
    preflight["realization_equal_weight_loss_check"] = _equal_weight_loss_check()
    preflight["seed_realization_pool_complete"] = len(realization_paths) == TARGET_EPISODES
    preflight["seed_realization_directories"] = [str(path.resolve()) for path in realization_paths]
    preflight["normalization_postclip_limit"] = max(abs(value) for value in TRIGGER_INPUT_CLIP)
    preflight["teacher_recomputed"] = False
    preflight["trigger_updates_during_collection"] = 0
    preflight["realizations_per_update"] = REALIZATIONS_PER_UPDATE
    return normalization, source, preflight


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return math.isfinite(float(value))
    return True


def _build_report(
    train_result: dict[str, Any], preflight: dict[str, Any], normalization_source: dict[str, Any]
) -> dict[str, Any]:
    update_records = _read_records(RUN_DIR / "update_metrics.jsonl", "update")
    episode_records = _read_records(RUN_DIR / "episode_metrics.jsonl", "episode")
    episodes_by_update: dict[int, list[dict[str, Any]]] = {}
    for episode_record in episode_records:
        episodes_by_update.setdefault(
            int(episode_record.get("synchronized_update", 0)), []
        ).append(episode_record)
    for update_record in update_records:
        group_episodes = episodes_by_update.get(
            int(update_record.get("synchronized_update", 0)), []
        )
        versions = [
            record.get("old_trigger_policy_version") for record in group_episodes
        ]
        update_versions = update_record.get("old_trigger_policy_versions", [])
        update_record["trigger_params_bitwise_frozen_during_collection"] = bool(
            len(group_episodes) == REALIZATIONS_PER_UPDATE
            and len(set(versions)) == 1
            and versions[0] is not None
            and len(update_versions) == REALIZATIONS_PER_UPDATE
            and len(set(update_versions)) == 1
            and update_versions[0] == versions[0]
        )
        cache_rates = [
            float(record["actual_cache_update_rate"])
            for record in group_episodes
            if record.get("actual_cache_update_rate") is not None
        ]
        update_record["cache_update_rate_by_realization"] = cache_rates
        update_record["cache_update_rate_mean"] = (
            float(np.mean(cache_rates))
            if len(cache_rates) == REALIZATIONS_PER_UPDATE
            else None
        )
    first5 = update_records[:5]
    last5 = update_records[5:10]
    realization_last5 = [
        record
        for record in episode_records
        if int(record.get("synchronized_update", 0)) >= 6
    ]
    realization_spearman_last5 = [
        float(record["realization_policy_delta_spearman"])
        for record in realization_last5
        if record.get("realization_policy_delta_spearman") is not None
    ]
    accepted_kl_means = [
        float(value)
        for update in update_records
        for value in update.get("trigger_epoch_kl_means", [])
    ]
    accepted_kl_maxima = [
        float(value)
        for update in update_records
        for value in update.get("trigger_epoch_kl_maxima", [])
    ]
    rejected_epochs = sum(int(item.get("trigger_epoch_rejected_count_this_update", 0)) for item in update_records)
    accepted_epochs = sum(int(item.get("trigger_epoch_accepted_count_this_update", 0)) for item in update_records)
    rollbacks = sum(int(item.get("trigger_rollback_count_this_update", 0)) for item in update_records)
    p_last = _mean(last5, "trigger_p_update_after")
    std_last = _mean(last5, "trigger_p_update_state_std")
    spearman_first5 = _mean(first5, "trigger_policy_delta_spearman")
    spearman_last5 = _mean(last5, "trigger_policy_delta_spearman")
    realization_spearman_mean = (
        float(np.mean(realization_spearman_last5)) if realization_spearman_last5 else None
    )
    realization_spearman_min = (
        float(np.min(realization_spearman_last5)) if realization_spearman_last5 else None
    )
    realization_spearman_std = (
        float(np.std(realization_spearman_last5)) if realization_spearman_last5 else None
    )
    normalization_maxima = [
        float(update["trigger_postclip_normalized_obs_abs_max"])
        for update in update_records
        if update.get("trigger_postclip_normalized_obs_abs_max") is not None
    ]
    normalization_stable = bool(
        preflight.get("pass")
        and preflight.get("normalization_unit_and_clip_check", {}).get("pass")
        and len(normalization_maxima) == len(update_records)
        and all(value <= 10.0 for value in normalization_maxima)
    )
    finite_records = all(_finite(item) for item in update_records + episode_records)
    accepted_kl_safe = bool(accepted_kl_means) and all(
        value <= HARD_KL for value in accepted_kl_means
    )
    collapse = bool(
        p_last is not None and not (0.05 < p_last < 0.95)
    )
    alignment_stable = bool(
        len(first5) == 5
        and len(last5) == 5
        and spearman_first5 is not None
        and spearman_last5 is not None
        and spearman_last5 >= spearman_first5 - 0.05
        and spearman_last5 >= 0.20
    )
    realization_alignment_ok = bool(
        len(realization_spearman_last5) == 5 * REALIZATIONS_PER_UPDATE
        and realization_spearman_mean is not None
        and realization_spearman_mean > 0.0
    )
    state_supported = bool(
        len(update_records) == TARGET_UPDATES
        and len(episode_records) == TARGET_EPISODES
        and not collapse
        and std_last is not None
        and std_last > 0.02
        and alignment_stable
        and realization_alignment_ok
        and accepted_kl_safe
        and normalization_stable
        and finite_records
        and int(train_result.get("completed_episodes", 0)) == TARGET_EPISODES
    )
    implementation_complete = bool(
        len(update_records) == TARGET_UPDATES
        and len(episode_records) == TARGET_EPISODES
        and int(train_result.get("completed_episodes", 0)) == TARGET_EPISODES
        and finite_records
        and normalization_stable
        and accepted_kl_safe
        and int(
            (json.loads((RUN_DIR / "run_summary.json").read_text(encoding="utf-8"))
             .get("training_counters", {})
             .get("formal_non_trigger_optimizer_steps", -1))
        ) == 0
        if (RUN_DIR / "run_summary.json").is_file()
        else False
    )
    if state_supported and alignment_stable:
        failure_mode = "NONE"
    elif collapse:
        failure_mode = "TRIGGER_COLLAPSE"
    elif not accepted_kl_safe:
        failure_mode = "KL_INSTABILITY"
    elif std_last is not None and std_last <= 0.02:
        failure_mode = "STATE_DEPENDENCE_LOST"
    elif not alignment_stable or not realization_alignment_ok:
        failure_mode = "CROSS_REALIZATION_GENERALIZATION_REMAINS"
    elif not normalization_stable:
        failure_mode = "NORMALIZATION_UNSTABLE"
    elif not implementation_complete:
        failure_mode = "INCOMPLETE_RUN"
    else:
        failure_mode = "STATE_DEPENDENT_TRIGGER_LEARNING_NOT_SUPPORTED"
    total_non_trigger = 0
    run_summary_path = RUN_DIR / "run_summary.json"
    if run_summary_path.is_file():
        total_non_trigger = int(
            json.loads(run_summary_path.read_text(encoding="utf-8"))
            .get("training_counters", {})
            .get("formal_non_trigger_optimizer_steps", -1)
        )
    return {
        "experiment": "V15-C4 multi-realization synchronized Trigger training gate",
        "status": "COMPLETED" if implementation_complete else "IN_PROGRESS",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "base_seed": BASE_SEED,
        "episode_seed_start": EPISODE_SEED_START,
        "episode_seed_end": EPISODE_SEED_START + TARGET_EPISODES - 1,
        "target_episodes": TARGET_EPISODES,
        "target_synchronized_updates": TARGET_UPDATES,
        "realizations_per_update": REALIZATIONS_PER_UPDATE,
        "episodes_completed": int(train_result.get("completed_episodes", len(episode_records))),
        "synchronized_updates_completed": len(update_records),
        "normalization_source": normalization_source,
        "normalization_protocol": {
            "source_sha256": EXPECTED_NORMALIZATION_SOURCE_SHA256,
            "raw_std_threshold": 1.0e-6,
            "low_variance_scale": 1.0,
            "other_feature_scale": "raw_std",
            "input_clip": TRIGGER_INPUT_CLIP,
        },
        "preflight": preflight,
        "v15c3_baseline": {
            "policy_delta_spearman_first10_mean": 0.335827,
            "policy_delta_spearman_last10_mean": 0.051515,
            "p_update_state_std_last10_mean": 0.261841,
        },
        "synchronized_updates": update_records,
        "realizations": episode_records,
        "accepted_kl_mean_values": accepted_kl_means,
        "accepted_kl_per_state_max_values": accepted_kl_maxima,
        "accepted_kl_max": max(accepted_kl_means) if accepted_kl_means else None,
        "accepted_kl_mean": float(np.mean(accepted_kl_means)) if accepted_kl_means else None,
        "accepted_epoch_count": accepted_epochs,
        "rejected_epoch_count": rejected_epochs,
        "rollback_count": rollbacks,
        "non_trigger_optimizer_steps": total_non_trigger,
        "normalization_stable": normalization_stable,
        "nan_inf_free": finite_records,
        "implementation_complete": "YES" if implementation_complete else "NO",
        "IMPLEMENTATION_COMPLETE": "YES" if implementation_complete else "NO",
        "SYNC_UPDATES_COMPLETED": len(update_records),
        "EPISODES_COMPLETED": len(episode_records),
        "REALIZATIONS_PER_UPDATE": REALIZATIONS_PER_UPDATE,
        "NON_TRIGGER_OPTIMIZER_STEPS": total_non_trigger,
        "RETURN_FIRST5_UPDATE_MEAN": _mean(first5, "return_mean_of_realizations"),
        "RETURN_LAST5_UPDATE_MEAN": _mean(last5, "return_mean_of_realizations"),
        "P_UPDATE_FIRST5_MEAN": _mean(first5, "trigger_p_update_after"),
        "P_UPDATE_LAST5_MEAN": p_last,
        "P_UPDATE_STATE_STD_FIRST5_MEAN": _mean(first5, "trigger_p_update_state_std"),
        "P_UPDATE_STATE_STD_LAST5_MEAN": std_last,
        "POLICY_DELTA_SPEARMAN_FIRST5_MEAN": spearman_first5,
        "POLICY_DELTA_SPEARMAN_LAST5_MEAN": spearman_last5,
        "REALIZATION_SPEARMAN_LAST5_MEAN": realization_spearman_mean,
        "REALIZATION_SPEARMAN_LAST5_MIN": realization_spearman_min,
        "REALIZATION_SPEARMAN_LAST5_STD": realization_spearman_std,
        "TRIGGER_ENTROPY_FIRST5_MEAN": _mean(first5, "trigger_entropy"),
        "TRIGGER_ENTROPY_LAST5_MEAN": _mean(last5, "trigger_entropy"),
        "CACHE_UPDATE_RATE_MEAN": _mean(update_records, "cache_update_rate_mean"),
        "ACTIVE_FRACTION_MEAN": _mean(update_records, "trigger_credit_active_fraction"),
        "DELTA_POSITIVE_FRACTION_MEAN": _mean(update_records, "trigger_delta_positive_fraction"),
        "DELTA_NEGATIVE_FRACTION_MEAN": _mean(update_records, "trigger_delta_negative_fraction"),
        "SYNC_BATCH_RAW_DELTA_MAD": [item.get("sync_batch_raw_delta_mad") for item in update_records],
        "ACCEPTED_KL_MAX": max(accepted_kl_means) if accepted_kl_means else None,
        "ACCEPTED_KL_MEAN": float(np.mean(accepted_kl_means)) if accepted_kl_means else None,
        "TOTAL_ACCEPTED_EPOCHS": accepted_epochs,
        "TOTAL_REJECTED_EPOCHS": rejected_epochs,
        "TOTAL_ROLLBACKS": rollbacks,
        "NORMALIZATION_STABLE": "YES" if normalization_stable else "NO",
        "TRIGGER_COLLAPSE": "YES" if collapse else "NO",
        "CROSS_REALIZATION_ALIGNMENT_STABLE": "YES" if alignment_stable else "NO",
        "MULTI_REALIZATION_REPAIR_SUPPORTED": "YES" if state_supported else "NO",
        "STATE_DEPENDENT_TRIGGER_LEARNING_SUPPORTED": "YES" if state_supported else "NO",
        "READY_FOR_STAGE2": "YES" if state_supported else "NO",
        "PRIMARY_FAILURE_MODE": failure_mode,
        "next_step": (
            "Use the same synchronized training structure to gradually unfreeze Candidate and Duration."
            if state_supported
            else "Stop; change the experiment plan before any Stage-2 training."
        ),
    }


def main() -> int:
    c2 = _load_module("v15c2_normalization_gate", C2_RUNNER_PATH)
    c3 = _load_module("v15c3_trigger_gate", C3_RUNNER_PATH)
    normalization, normalization_source, preflight = _validate_inputs_and_preflight(c2, c3)
    if not preflight.get("pass"):
        raise RuntimeError("V15-C4 preflight failed.")
    run_dir_resolved = RUN_DIR.resolve(strict=False)
    runs_root = (PROJECT_ROOT / "runs").resolve(strict=False)
    if run_dir_resolved.parent != runs_root:
        raise ValueError(f"C4 run directory must be a direct child of runs/: {run_dir_resolved}")
    if run_dir_resolved.exists() and (not run_dir_resolved.is_dir() or any(run_dir_resolved.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty C4 run directory: {run_dir_resolved}")
    initial_report = {
        "experiment": "V15-C4 multi-realization synchronized Trigger training gate",
        "status": "PREFLIGHT_PASSED",
        "formal_runner_path": str(Path(__file__).resolve()),
        "formal_command": FORMAL_COMMAND,
        "run_dir": str(RUN_DIR.resolve()),
        "report_path": str(REPORT_PATH.resolve()),
        "target_episodes": TARGET_EPISODES,
        "target_synchronized_updates": TARGET_UPDATES,
        "realizations_per_update": REALIZATIONS_PER_UPDATE,
        "normalization_source": normalization_source,
        "preflight": preflight,
    }
    _write_json(REPORT_PATH, initial_report)

    from ppo_simple.trainer import run_synchronized_trigger_training

    def save_progress(update_index: int, completed_episodes: int) -> None:
        progress = _build_report(
            {
                "completed_episodes": completed_episodes,
                "synchronized_updates_completed": update_index,
            },
            preflight,
            normalization_source,
        )
        progress["status"] = "IN_PROGRESS"
        _write_json(REPORT_PATH, progress)

    train_result = run_synchronized_trigger_training(
        CONFIG_PATH,
        TARGET_EPISODES,
        RUN_DIR,
        episode_pool=EPISODE_POOL,
        episode_seed_start=EPISODE_SEED_START,
        realizations_per_update=REALIZATIONS_PER_UPDATE,
        macro_actor_learning_rate_override=MACRO_ACTOR_LR,
        seed=BASE_SEED,
        initial_checkpoint_path=V14B_CHECKPOINT,
        expected_initial_checkpoint_sha256=V14B_SHA256,
        trigger_observation_normalization=normalization,
        progress_callback=save_progress,
    )
    result = _build_report(train_result, preflight, normalization_source)
    _write_json(REPORT_PATH, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "IMPLEMENTATION_COMPLETE",
                    "SYNC_UPDATES_COMPLETED",
                    "EPISODES_COMPLETED",
                    "RETURN_FIRST5_UPDATE_MEAN",
                    "RETURN_LAST5_UPDATE_MEAN",
                    "POLICY_DELTA_SPEARMAN_FIRST5_MEAN",
                    "POLICY_DELTA_SPEARMAN_LAST5_MEAN",
                    "REALIZATION_SPEARMAN_LAST5_MEAN",
                    "REALIZATION_SPEARMAN_LAST5_MIN",
                    "ACCEPTED_KL_MAX",
                    "NORMALIZATION_STABLE",
                    "TRIGGER_COLLAPSE",
                    "CROSS_REALIZATION_ALIGNMENT_STABLE",
                    "MULTI_REALIZATION_REPAIR_SUPPORTED",
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
