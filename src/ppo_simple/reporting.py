"""Run provenance and compact summary reporting for PPO experiments."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from common import project_config

from .plotting import (
    TRAINING_EPOCH_RETURN_PLOT_FILENAME,
    TRAINING_SLOT_REWARD_PLOT_FILENAME,
    _load_slot_contract,
    _read_slot_metrics,
)


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
FRESH_RUN_ARTIFACTS = (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    "episode_metrics.jsonl",
    "update_metrics.jsonl",
    "critic_diagnostics.jsonl",
    "slot_metrics.jsonl",
    "latest_checkpoint.pt",
    "best_train_return_checkpoint.pt",
    f"plots/{TRAINING_EPOCH_RETURN_PLOT_FILENAME}",
    f"plots/{TRAINING_SLOT_REWARD_PLOT_FILENAME}",
)


def ensure_fresh_run_dir_is_clean(run_dir: Path) -> None:
    """Reject fresh runs that would mix with an existing experiment."""

    run_path = Path(run_dir)
    if not run_path.exists():
        return
    if not run_path.is_dir():
        raise RuntimeError(
            "fresh PPO run-dir already exists but is not a directory; "
            "use a new --run-dir or --resume"
        )
    existing_artifacts = [
        relative_path
        for relative_path in FRESH_RUN_ARTIFACTS
        if (run_path / relative_path).exists()
    ]
    if existing_artifacts:
        raise RuntimeError(
            "fresh PPO run-dir already contains experiment artifacts; "
            "use a new --run-dir or --resume"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_parse_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError(f"reporting JSON root must be an object: {path}")
    return payload


def _parse_nonfinite(value: str) -> float:
    return float(value)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    nan_inf_detected = False
    if not path.exists():
        return records, nan_inf_detected
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"reporting JSONL contains a blank line: {path}:{line_number}")
            try:
                record = json.loads(line, parse_constant=_parse_nonfinite)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"reporting JSONL contains invalid JSON: {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"reporting JSONL line must be an object: {path}:{line_number}"
                )
            nan_inf_detected = nan_inf_detected or _contains_nonfinite(record)
            records.append(record)
    return records, nan_inf_detected


def _finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reporting field {field} must be numeric or null") from exc
    return converted if math.isfinite(converted) else None


def _git_identity(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if sha_result.returncode != 0 or dirty_result.returncode != 0:
            raise RuntimeError(
                f"git command failed: sha={sha_result.returncode}, "
                f"dirty={dirty_result.returncode}"
            )
        sha = sha_result.stdout.strip()
        if not sha:
            raise RuntimeError("git rev-parse HEAD returned an empty SHA")
        return sha, bool(dirty_result.stdout.strip())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[REPORTING] Warning: unable to read git identity: {exc}", flush=True)
        return None, None


def _optimizer_learning_rate(optimizer: Any, label: str) -> float:
    if optimizer is None:
        raise ValueError(f"{label} optimizer is not initialized")
    groups = list(getattr(optimizer, "param_groups", ()))
    if not groups:
        raise ValueError(f"{label} optimizer has no parameter groups")
    rates = []
    for group in groups:
        rate = float(group["lr"])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{label} optimizer learning rate is invalid: {rate}")
        rates.append(rate)
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError(
            f"{label} optimizer has multiple learning rates; manifest field is scalar"
        )
    return rates[0]


def _actual_optimizers(trainer: Any) -> tuple[Any, Any]:
    actor_optimizer = getattr(trainer, "actor_optimizer", None)
    critic_optimizer = getattr(trainer, "critic_optimizer", None)
    shared_optimizer = getattr(trainer, "optimizer", None)
    if actor_optimizer is None:
        actor_optimizer = shared_optimizer
    if critic_optimizer is None:
        critic_optimizer = shared_optimizer
    return actor_optimizer, critic_optimizer


def build_run_manifest(
    *,
    config_path: str | Path,
    config_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_epochs: int,
    rollout_steps: int,
) -> dict[str, Any]:
    """Build a manifest from the live trainer, policy, optimizers, and config."""

    get = project_config.get_config_value
    hyper = trainer.hyperparameters
    actor_optimizer, critic_optimizer = _actual_optimizers(trainer)
    protocol = project_config.get_ppo_data_protocol(context.cfg)
    train_origin_start = int(context.train_origin_start)
    train_origin_end = int(context.train_origin_end)
    test_origin_start = protocol["test_origin_start"]
    test_origin_end = protocol["test_origin_end"]
    policy_schema = str(getattr(trainer.policy, "architecture_id", context.policy_schema))
    is_proposed_joint = policy_schema == "adaptive_joint_cache_uav_ppo"
    git_sha, git_dirty = _git_identity(Path(context.root))
    resolved_config_path = Path(
        context.cfg.get("_meta", {}).get("config_path", config_path)
    ).resolve()
    learning_rate_metadata = trainer.learning_rate_metadata()
    training_provenance = context.cfg.get("training_provenance", {})
    q_cache_manifest_fields: dict[str, Any] = {}
    if getattr(trainer, "q_cache_optimizer", None) is not None:
        q_cache_manifest_fields = {
            "cache_credit_granularity": "FULL_NODE_ACTION",
            "q_cache_learning_rate": _optimizer_learning_rate(
                trainer.q_cache_optimizer, "q_cache"
            ),
            "q_cache_target_source": "current_PopArt_normalized_raw_GAE_return",
            "q_cache_conditioning": (
                "full_state_plus_target_cache_node_plus_full_cache_and_uav_action_encoding"
            ),
            "q_cache_action_encoding": (
                "per_node_operation_one_hot_plus_candidate_one_hot_plus_production_primitive_feature"
            ),
            "q_cache_optimizer_steps_per_rollout": int(
                hyper.update_epochs * hyper.num_minibatches
            ),
            "cache_counterfactual_advantage_source": (
                "detached_Q_selected_minus_legal_action_policy_expectation"
            ),
        }
    cache_surrogate_manifest_fields: dict[str, Any] = {}
    if hasattr(trainer, "cache_surrogate_metadata"):
        cache_surrogate_manifest_fields = dict(trainer.cache_surrogate_metadata())
    return {
        "experiment_type": "main_training",
        "architecture_version": int(getattr(trainer.policy, "architecture_version", 1)),
        "initial_actor_sha256": training_provenance.get("ACTOR_INITIAL_STATE_SHA256"),
        "initial_critic_sha256": training_provenance.get("CRITIC_INITIAL_STATE_SHA256"),
        "initial_uav_candidate_encoder_sha256": training_provenance.get(
            "INITIAL_UAV_CANDIDATE_ENCODER_SHA256"
        ),
        "uav_candidate_encoder_initialization": (
            "deterministic deepcopy of the seeded base UAV encoder before optimization"
            if is_proposed_joint else None
        ),
        "checkpoint_interval_epochs": 10 if is_proposed_joint else None,
        "best_train_return_checkpoint_role": "TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION",
        "final_checkpoint_selection_metric": "validation_mean_episode_return" if is_proposed_joint else None,
        "git_commit_sha": git_sha,
        "git_dirty": git_dirty,
        "config_path": str(resolved_config_path),
        "config_sha256": str(config_sha256),
        "seed": int(seed),
        "device": str(trainer.device),
        "policy_schema": policy_schema,
        "configuration_control_mode": str(context.configuration_control_mode),
        "advantage_normalization_mode": str(hyper.advantage_normalization_mode),
        "operation_advantage_normalization_mode": str(
            hyper.operation_advantage_normalization_mode
        ),
        "operation_advantage_reference_scale": float(
            hyper.operation_advantage_reference_scale
        ),
        "uav_operation_advantage_source": (
            "FULL_INFORMATION_EXACT_Q_TEACHER" if is_proposed_joint else "GLOBAL_GAE"
        ),
        "UAV_OPERATION_TEACHER": (
            "FULL_INFORMATION_EXACT_Q_TEACHER" if is_proposed_joint else "GLOBAL_GAE"
        ),
        "uav_candidate_advantage_source": (
            "EXACT_Q_LISTWISE"
            if is_proposed_joint
            else "GLOBAL_GAE"
        ),
        "UAV_CANDIDATE_TEACHER": (
            "EXACT_Q_LISTWISE" if is_proposed_joint else "GLOBAL_GAE"
        ),
        "UAV_TEACHER_UNCHANGED": "YES" if is_proposed_joint else "N/A",
        "candidate_advantage_normalization": (
            "EXACT_Q_LISTWISE"
            if is_proposed_joint
            else "LEGACY_GLOBAL_GAE"
        ),
        "cache_trigger_advantage_source": str(
            getattr(
                trainer,
                "cache_trigger_advantage_source",
                project_config.get_config_value(
                    context.cfg,
                    "algorithm.ppo.cache_trigger_advantage_source",
                    "GLOBAL_GAE",
                ),
            )
        ),
        "cache_candidate_advantage_source": str(
            getattr(
                trainer,
                "cache_candidate_advantage_source",
                project_config.get_config_value(
                    context.cfg,
                    "algorithm.ppo.cache_candidate_advantage_source",
                    "GLOBAL_GAE",
                ),
            )
        ),
        "cache_trigger_ratio_granularity": str(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.cache_trigger_ratio_granularity",
                "WHOLE_CACHE_JOINT",
            )
        ),
        "cache_candidate_ratio_granularity": str(
            project_config.get_config_value(
                context.cfg,
                "algorithm.ppo.cache_candidate_ratio_granularity",
                "WHOLE_CACHE_JOINT",
            )
        ),
        "cache_surrogate_reduction": "NOT_USED_FOR_PRODUCTION_CREDIT",
        "CACHE_POLICY_CREDIT_SOURCE": "ACTUAL_ROLLOUT_GAE",
        "CACHE_ACTION_Q_CRITIC_INITIALIZED": bool(
            getattr(trainer, "cache_action_q_critic", None) is not None
        ),
        "CACHE_ACTION_Q_USED_FOR_POLICY_CREDIT": "NO",
        "CACHE_COUNTERFACTUAL_MODE": "NONE_FOR_PRODUCTION",
        "CACHE_COUNTERFACTUAL_ENV_EXECUTIONS": 0,
        "critic_value_scale_mode": str(hyper.critic_value_scale_mode),
        "critic_value_scale_epsilon": float(hyper.critic_value_scale_epsilon),
        "critic_value_scale_warmup_samples": int(hyper.critic_value_scale_warmup_samples),
        "observation_dim": int(context.observation_spec.observation_dim),
        "action_dim": int(context.action_spec.action_dim),
        "data_protocol": str(protocol["protocol"]),
        "episode_slots": int(context.episode_slots),
        "origin_start": train_origin_start,
        "origin_end": train_origin_end,
        "terminal_origin": int(context.terminal_origin),
        "train_origin_start": train_origin_start,
        "train_origin_end": train_origin_end,
        "test_origin_start": (
            None if test_origin_start is None else int(test_origin_start)
        ),
        "test_origin_end": None if test_origin_end is None else int(test_origin_end),
        "target_epochs": int(target_epochs),
        "rollout_steps": int(rollout_steps),
        "origins_per_epoch": int(context.episode_slots),
        "gamma": float(hyper.gamma),
        "gae_lambda": float(hyper.gae_lambda),
        "clip_epsilon": float(hyper.clip_epsilon),
        "update_epochs": int(hyper.update_epochs),
        "target_kl": hyper.target_kl,
        "cache_kl_control": str(
            getattr(trainer, "cache_kl_control", "epoch")
        ),
        "num_minibatches": int(hyper.num_minibatches),
        "minibatch_size": int(context.minibatch_size),
        "actor_learning_rate": _optimizer_learning_rate(
            actor_optimizer, "actor"
        ),
        "critic_learning_rate": _optimizer_learning_rate(
            critic_optimizer, "critic"
        ),
        "max_grad_norm": float(hyper.max_grad_norm),
        "entropy_coefficient": float(hyper.entropy_coefficient),
        "candidate_policy_improvement": str(
            getattr(trainer, "candidate_policy_improvement", "LEGACY_PPO")
        ),
        "exact_q_listwise_temperature": float(
            getattr(trainer, "exact_q_listwise_temperature", 1.0)
        ),
        **learning_rate_metadata,
        **q_cache_manifest_fields,
        **cache_surrogate_manifest_fields,
    }


def _manifest_protocol_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize protocol metadata while accepting old legacy manifests."""

    protocol = str(
        manifest.get(
            "data_protocol",
            project_config.PPO_DATA_PROTOCOL_LEGACY_SPLIT,
        )
    )
    if protocol not in {
        project_config.PPO_DATA_PROTOCOL_LEGACY_SPLIT,
        project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED,
    }:
        raise ValueError(f"unknown PPO run_manifest data_protocol: {protocol}")
    origin_start = int(
        manifest.get("origin_start", manifest.get("train_origin_start", 0))
    )
    raw_origin_end = manifest.get(
        "origin_end", manifest.get("train_origin_end")
    )
    raw_episode_slots = manifest.get(
        "episode_slots", manifest.get("origins_per_epoch")
    )
    if raw_origin_end is None and raw_episode_slots is not None:
        raw_origin_end = origin_start + int(raw_episode_slots) - 1
    if raw_origin_end is None:
        raise ValueError("PPO run_manifest is missing origin_end/episode_slots")
    origin_end = int(raw_origin_end)
    episode_slots = int(
        raw_episode_slots
        if raw_episode_slots is not None
        else origin_end - origin_start + 1
    )
    if episode_slots <= 0 or origin_end - origin_start + 1 != episode_slots:
        raise ValueError("PPO run_manifest has inconsistent episode slot bounds")
    terminal_origin = int(manifest.get("terminal_origin", origin_end))
    if terminal_origin != origin_end:
        raise ValueError("PPO run_manifest terminal_origin must equal origin_end")

    raw_rollout_steps = manifest.get("rollout_steps")
    rollout_steps = None if raw_rollout_steps is None else int(raw_rollout_steps)
    raw_num_minibatches = manifest.get("num_minibatches")
    num_minibatches = (
        None if raw_num_minibatches is None else int(raw_num_minibatches)
    )
    raw_minibatch_size = manifest.get("minibatch_size")
    if raw_minibatch_size is None and num_minibatches is not None:
        minibatch_size = (
            episode_slots // num_minibatches
            if num_minibatches > 0 and episode_slots % num_minibatches == 0
            else None
        )
    else:
        minibatch_size = (
            None if raw_minibatch_size is None else int(raw_minibatch_size)
        )
    return {
        "data_protocol": protocol,
        "episode_slots": episode_slots,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "terminal_origin": terminal_origin,
        "rollout_steps": rollout_steps,
        "num_minibatches": num_minibatches,
        "minibatch_size": minibatch_size,
    }


def validate_run_manifest_for_resume(
    existing: dict[str, Any], current: dict[str, Any]
) -> None:
    """Reject a resume when immutable experiment identity differs."""

    required_match_fields = (
        "config_sha256",
        "policy_schema",
        "advantage_normalization_mode",
        "critic_value_scale_mode",
        "observation_dim",
        "rollout_steps",
        "experiment_id",
        "control_mode",
        "configured_actor_learning_rate",
        "configured_critic_learning_rate",
        "effective_actor_learning_rate",
        "effective_critic_learning_rate",
        "actor_learning_rate_overridden",
        "critic_learning_rate_overridden",
        "lr_experiment_axis",
        "lr_experiment_value",
    )
    mismatches = {
        field: (existing.get(field), current.get(field))
        for field in required_match_fields
        if existing.get(field) != current.get(field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in mismatches.items()
        )
        raise ValueError(f"PPO resume run_manifest mismatch: {details}")

    existing_protocol = _manifest_protocol_identity(existing)
    current_protocol = _manifest_protocol_identity(current)
    protocol_mismatches = {
        field: (existing_protocol[field], current_protocol[field])
        for field in (
            "data_protocol",
            "episode_slots",
            "origin_start",
            "origin_end",
            "terminal_origin",
            "rollout_steps",
            "num_minibatches",
            "minibatch_size",
        )
        if existing_protocol[field] != current_protocol[field]
    }
    if protocol_mismatches:
        details = ", ".join(
            f"{field}: existing={old!r}, current={new!r}"
            for field, (old, new) in protocol_mismatches.items()
        )
        raise ValueError(f"PPO resume protocol mismatch: {details}")

    existing_git_sha = existing.get("git_commit_sha")
    current_git_sha = current.get("git_commit_sha")
    if existing_git_sha is not None:
        existing_git_sha = str(existing_git_sha).strip() or None
    if current_git_sha is not None:
        current_git_sha = str(current_git_sha).strip() or None
    if (
        existing_git_sha is not None
        and current_git_sha is not None
        and existing_git_sha != current_git_sha
    ):
        raise ValueError(
            "PPO resume git_commit_sha mismatch; refusing cross-code-version resume"
        )
    if existing_git_sha is None or current_git_sha is None:
        print(
            "[REPORTING] Warning: Git provenance unavailable for resume; "
            "allowing resume without cross-code-version verification.",
            flush=True,
        )


def write_run_manifest(
    path: Path,
    *,
    config_path: str | Path,
    config_sha256: str,
    seed: int,
    trainer: Any,
    context: Any,
    target_epochs: int,
    rollout_steps: int,
    resume: bool,
) -> dict[str, Any]:
    """Create a fresh manifest or validate/update one for a resume."""

    current = build_run_manifest(
        config_path=config_path,
        config_sha256=config_sha256,
        seed=seed,
        trainer=trainer,
        context=context,
        target_epochs=target_epochs,
        rollout_steps=rollout_steps,
    )
    if resume and path.exists():
        existing = _read_json(path)
        validate_run_manifest_for_resume(existing, current)
        initial_target_epochs = int(
            existing.get("initial_target_epochs", existing.get("target_epochs", target_epochs))
        )
        merged = dict(existing)
        merged["initial_target_epochs"] = initial_target_epochs
        merged["current_target_epochs"] = int(target_epochs)
        merged["target_epochs"] = int(target_epochs)
        _write_json(path, merged)
        return merged

    current["initial_target_epochs"] = int(target_epochs)
    current["current_target_epochs"] = int(target_epochs)
    _write_json(path, current)
    return current


def _episode_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    raw_records, nan_inf_detected = _read_jsonl(path)
    by_epoch: dict[int, dict[str, Any]] = {}
    for record in raw_records:
        if record.get("record_type", "episode") != "episode":
            continue
        try:
            epoch = int(record["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"episode_metrics record has invalid epoch: {record}") from exc
        if epoch < 1:
            raise ValueError(f"episode_metrics epoch must be >=1: {epoch}")
        if epoch in by_epoch:
            raise ValueError(f"episode_metrics duplicate epoch: {epoch}")
        by_epoch[epoch] = record
    ordered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    return ordered, nan_inf_detected


def _metric_values(
    records: list[dict[str, Any]], field: str, *, allow_null: bool = True
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _finite_float(record.get(field), field)
        if value is None:
            if not allow_null:
                raise ValueError(f"episode_metrics field {field} is missing or null")
            continue
        values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rolling_mean(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, float | None]:
    if not returns:
        return None, None
    actual_window = min(int(window), len(returns))
    return float(sum(returns[-actual_window:]) / actual_window), None


def _rolling_max(
    returns: list[float], epochs: list[int], window: int
) -> tuple[float | None, int | None]:
    if len(returns) < int(window):
        return None, None
    candidates = [
        (
            float(sum(returns[index - window + 1 : index + 1]) / window),
            int(epochs[index]),
        )
        for index in range(window - 1, len(returns))
    ]
    best = max(candidates, key=lambda item: item[0])
    return best


def update_run_summary(
    path: Path,
    *,
    episode_metrics_path: Path,
    slot_metrics_path: Path,
    completed_epochs: int,
    target_epochs: int,
    status: str,
    protocol_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the current compact summary from already-written metrics."""

    if status not in {"running", "completed"}:
        raise ValueError(f"invalid run summary status: {status}")
    episodes, episode_nan_inf = _episode_records(episode_metrics_path)
    if protocol_metadata is None:
        episode_slots, origin_start = _load_slot_contract(slot_metrics_path.parent)
        protocol_metadata = {
            "data_protocol": project_config.PPO_DATA_PROTOCOL_LEGACY_SPLIT,
            "episode_slots": episode_slots,
            "origin_start": origin_start,
            "origin_end": origin_start + episode_slots - 1,
            "terminal_origin": origin_start + episode_slots - 1,
        }
    episode_slots = int(protocol_metadata["episode_slots"])
    origin_start = int(protocol_metadata["origin_start"])
    origin_end = int(protocol_metadata["origin_end"])
    terminal_origin = int(protocol_metadata.get("terminal_origin", origin_end))
    if origin_end - origin_start + 1 != episode_slots:
        raise ValueError("run summary protocol has inconsistent episode slot bounds")
    if terminal_origin != origin_end:
        raise ValueError("run summary terminal_origin must equal origin_end")
    slot_records = _read_slot_metrics(
        slot_metrics_path,
        episode_slots=episode_slots,
        origin_start=origin_start,
    )
    epochs = [int(record["epoch"]) for record in episodes]
    returns = [
        value
        for value in (_finite_float(record.get("episode_return"), "episode_return") for record in episodes)
        if value is not None
    ]
    if len(returns) != len(episodes):
        episode_nan_inf = True
    final_window_count = min(20, len(episodes))
    final_episodes = episodes[-final_window_count:] if final_window_count else []
    final_epoch = epochs[-1] if epochs else None
    final_return = returns[-1] if returns else None
    best_epoch = 0
    best_return: float | None = None
    if returns:
        best_index = max(range(len(returns)), key=lambda index: returns[index])
        best_epoch = epochs[best_index]
        best_return = returns[best_index]

    ma20, _ = _rolling_mean(returns, epochs, 20)
    ma50, _ = _rolling_mean(returns, epochs, 50)
    ma20_max, ma20_max_epoch = _rolling_max(returns, epochs, 20)
    ma50_max, ma50_max_epoch = _rolling_max(returns, epochs, 50)

    critic_ev = _metric_values(final_episodes, "critic_explained_variance")
    all_critic_ev = _metric_values(episodes, "critic_explained_variance")
    max_critic_ev = None
    max_critic_ev_epoch = None
    if all_critic_ev:
        max_index = max(
            (index for index, record in enumerate(episodes)
             if _finite_float(record.get("critic_explained_variance"), "critic_explained_variance") is not None),
            key=lambda index: float(episodes[index]["critic_explained_variance"]),
        )
        max_critic_ev = float(episodes[max_index]["critic_explained_variance"])
        max_critic_ev_epoch = int(episodes[max_index]["epoch"])

    final_epoch_set = {int(record["epoch"]) for record in final_episodes}
    final_slot_records = [
        record
        for (epoch, _), record in slot_records.items()
        if epoch in final_epoch_set
    ]

    def region_mean(start: int, end: int) -> float | None:
        values = [
            float(record["reward"])
            for record in final_slot_records
            if start <= int(record["origin"]) <= end
        ]
        return _mean(values)

    base_region_size = episode_slots // 4
    region_lengths = [base_region_size] * 3 + [
        episode_slots - (base_region_size * 3)
    ]
    region_values: dict[str, float | None] = {}
    region_start = origin_start
    for region_length in region_lengths:
        region_end = region_start + region_length - 1
        if region_length <= 0:
            value = None
        else:
            value = region_mean(region_start, region_end)
        region_values[
            f"final20_reward_origin_{region_start}_{region_end}"
        ] = value
        region_start = region_end + 1

    summary = {
        "status": status,
        "completed_epochs": int(completed_epochs),
        "target_epochs": int(target_epochs),
        "data_protocol": str(protocol_metadata["data_protocol"]),
        "episode_slots": episode_slots,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "terminal_origin": terminal_origin,
        "rollout_steps": protocol_metadata.get("rollout_steps"),
        "num_minibatches": protocol_metadata.get("num_minibatches"),
        "minibatch_size": protocol_metadata.get("minibatch_size"),
        "best_epoch": int(best_epoch),
        "best_return": best_return,
        "final_epoch": final_epoch,
        "final_return": final_return,
        "final20_window_count": int(final_window_count),
        "final20_mean_return": _mean(returns[-final_window_count:]) if final_window_count else None,
        "ma20": ma20,
        "ma20_max": ma20_max,
        "ma20_max_epoch": ma20_max_epoch,
        "ma50": ma50,
        "ma50_max": ma50_max,
        "ma50_max_epoch": ma50_max_epoch,
        "final20_mean_critic_ev": _mean(critic_ev),
        "max_critic_ev": max_critic_ev,
        "max_critic_ev_epoch": max_critic_ev_epoch,
        "final20_cache_change_rate": _mean(
            _metric_values(final_episodes, "cache_actual_change_rate")
        ),
        "final20_uav_movement_rate": _mean(
            _metric_values(final_episodes, "uav_actual_movement_rate")
        ),
        "nan_inf_detected": bool(episode_nan_inf),
    }
    summary.update(region_values)
    _write_json(path, summary)
    return summary
