from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import trainer as training
from .interface import load_environment_context
from .rollout import collect_dual_timescale_episode


V16A_RUN_ID = "V16A_DIRECT_TRIGGER_DISTILLATION_GATE"
REALIZATIONS_PER_UPDATE = 4
SYNC_UPDATES = 5
TARGET_EPISODES = REALIZATIONS_PER_UPDATE * SYNC_UPDATES
DELTA_HUBER_BETA = 1.0
DISTILLATION_MAX_EPOCHS = 10
DISTILLATION_LR = 1.0e-3
DISTILLATION_MAX_GRAD_NORM = 0.5
DISTILLATION_KL_LIMIT = 0.05


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Refusing to write NaN/Inf to a V16-A artifact.")
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return math.isfinite(float(value))
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _jsonable(payload),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _module_sha256(module: Any, *, exclude_trigger: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if exclude_trigger and name.startswith("macro_cache_trigger_actor."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_is_finite(module: Any) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in module.state_dict().values())


def _trigger_logit(actor: Any, normalized_observations: torch.Tensor) -> torch.Tensor:
    logits = actor(normalized_observations)
    if logits.ndim != 2 or int(logits.shape[1]) != 2:
        raise ValueError("V16-A Trigger must emit KEEP/UPDATE logits.")
    return logits[:, 1] - logits[:, 0]


def _trigger_probabilities(actor: Any, normalized_observations: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(_trigger_logit(actor, normalized_observations))


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("MAD requires at least one finite active DELTA.")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return training._spearman_correlation(left, right)


def _equal_weight_check() -> dict[str, Any]:
    values = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64, requires_grad=True)
    realization_ids = np.asarray([10, 10, 11], dtype=np.int64)
    loss = training._equal_realization_tensor_mean(values, realization_ids)
    loss.backward()
    expected_gradient = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    if not math.isclose(float(loss.detach()), 4.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError("Realization-equal mean loss check failed.")
    if not torch.allclose(values.grad, expected_gradient, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Realization-equal gradients check failed.")
    return {"pass": True, "loss": float(loss.detach()), "gradients": values.grad.tolist()}


def _target_and_huber_check() -> dict[str, Any]:
    delta = torch.tensor([2.0, -2.0], dtype=torch.float64)
    target = torch.clamp(delta / 1.0, -5.0, 5.0)
    if not (float(target[0]) > 0.0 and float(target[1]) < 0.0):
        raise AssertionError("Continuous target signs disagree with DELTA signs.")
    prediction = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    loss = F.smooth_l1_loss(prediction, target, beta=DELTA_HUBER_BETA, reduction="mean")
    loss.backward()
    if not (float(prediction.grad[0]) < 0.0 and float(prediction.grad[1]) > 0.0):
        raise AssertionError("Huber gradient does not move predictions toward target signs.")
    return {
        "pass": True,
        "target": target.tolist(),
        "prediction_gradient": prediction.grad.tolist(),
    }


def _transactional_epoch_step(
    actor: Any,
    optimizer: Any,
    *,
    replay_observations: torch.Tensor,
    active_observations: torch.Tensor,
    active_targets: torch.Tensor,
    active_realization_ids: np.ndarray,
    hard_kl: float,
) -> dict[str, Any]:
    snapshot = training._capture_transaction_state(actor, optimizer)
    with torch.no_grad():
        p_before = _trigger_probabilities(actor, replay_observations).detach()
    optimizer.zero_grad(set_to_none=True)
    predicted = _trigger_logit(actor, active_observations)
    per_sample_loss = F.smooth_l1_loss(
        predicted,
        active_targets,
        beta=DELTA_HUBER_BETA,
        reduction="none",
    )
    loss = training._equal_realization_tensor_mean(
        per_sample_loss, active_realization_ids
    )
    if not bool(torch.isfinite(loss).item()):
        training._restore_transaction_state(actor, optimizer, snapshot)
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("V16-A supervised Huber loss contains NaN/Inf.")
    loss.backward()
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, DISTILLATION_MAX_GRAD_NORM)
    if not bool(torch.isfinite(grad_norm).item()):
        training._restore_transaction_state(actor, optimizer, snapshot)
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("V16-A Trigger gradient norm contains NaN/Inf.")
    optimizer.step()
    with torch.no_grad():
        p_after = _trigger_probabilities(actor, replay_observations).detach()
        kl_values = training._binary_distribution_kl(p_before, p_after)
        mean_kl = float(kl_values.mean().item())
    if not math.isfinite(mean_kl):
        training._restore_transaction_state(actor, optimizer, snapshot)
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("V16-A policy KL contains NaN/Inf.")
    if mean_kl > float(hard_kl):
        training._restore_transaction_state(actor, optimizer, snapshot)
        optimizer.zero_grad(set_to_none=True)
        return {
            "accepted": False,
            "rolled_back": True,
            "huber_loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "mean_policy_kl": mean_kl,
        }
    return {
        "accepted": True,
        "rolled_back": False,
        "huber_loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm.detach().item()),
        "mean_policy_kl": mean_kl,
    }


def _transactional_rollback_check() -> dict[str, Any]:
    actor = torch.nn.Linear(2, 2, bias=True).to(dtype=torch.float64)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[0.2, -0.1], [-0.2, 0.1]], dtype=torch.float64))
        actor.bias.zero_()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.1)
    observations = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    targets = torch.tensor([5.0, -5.0], dtype=torch.float64)
    ids = np.asarray([1, 2], dtype=np.int64)
    original = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    result = _transactional_epoch_step(
        actor,
        optimizer,
        replay_observations=observations,
        active_observations=observations,
        active_targets=targets,
        active_realization_ids=ids,
        hard_kl=0.0,
    )
    restored = all(torch.equal(original[key], actor.state_dict()[key]) for key in original)
    if not (result["rolled_back"] and restored and not optimizer.state):
        raise AssertionError("V16-A transactional model/optimizer rollback check failed.")
    return {"pass": True, "mean_policy_kl": result["mean_policy_kl"], "restored": restored}


def _minimal_checks() -> dict[str, Any]:
    checks = {
        "continuous_target_sign_and_huber_gradient": _target_and_huber_check(),
        "realization_equal_weighting": _equal_weight_check(),
        "cumulative_replay_growth": {
            "pass": [4 * index for index in range(1, SYNC_UPDATES + 1)]
            == [4, 8, 12, 16, 20],
            "expected_realization_counts": [4, 8, 12, 16, 20],
        },
        "transactional_kl_rollback": _transactional_rollback_check(),
    }
    if not checks["cumulative_replay_growth"]["pass"]:
        raise AssertionError("Cumulative replay growth check failed.")
    return checks


def _collection_rows(
    collection: Any,
    *,
    policy: Any,
    episode_seed: int,
    realization_id: int,
) -> dict[str, np.ndarray]:
    transitions = collection.macro.transitions
    if not transitions:
        raise ValueError("V16-A rollout produced no Macro decisions.")
    raw_observations = torch.as_tensor(
        np.stack([np.asarray(item.observation, dtype=np.float32) for item in transitions]),
        dtype=torch.float32,
        device=next(policy.macro_cache_trigger_actor.parameters()).device,
    )
    normalized = policy._normalized_trigger_observations(raw_observations)
    normalized_np = normalized.detach().cpu().numpy().astype(np.float32, copy=True)
    if (
        normalized_np.shape != (len(transitions), 400)
        or not np.isfinite(normalized_np).all()
        or float(np.max(np.abs(normalized_np))) > 10.0
    ):
        raise FloatingPointError("Normalized Trigger replay observations are invalid.")
    deltas: list[float] = []
    standard_errors: list[float] = []
    active: list[bool] = []
    macro_indices: list[int] = []
    for macro_index, item in enumerate(transitions):
        valid = bool(item.trigger_learning_valid and item.trigger_teacher_valid)
        raw_delta = (
            0.0
            if item.teacher_delta_update_keep is None
            else float(item.teacher_delta_update_keep)
        )
        se_update = float(item.trigger_update_standard_error)
        if not (math.isfinite(raw_delta) and math.isfinite(se_update) and se_update >= 0.0):
            raise FloatingPointError("V16-A teacher DELTA/SE contains NaN/Inf or negative SE.")
        is_active = bool(valid and item.trigger_credit_active)
        expected_active = bool(valid and abs(raw_delta) > se_update)
        if is_active != expected_active:
            raise AssertionError("V16-A active mask differs from the unchanged uncertainty gate.")
        deltas.append(raw_delta)
        standard_errors.append(se_update)
        active.append(is_active)
        macro_indices.append(macro_index)
    count = len(transitions)
    rows = {
        "normalized_trigger_observation": normalized_np,
        "raw_delta": np.asarray(deltas, dtype=np.float64),
        "se_update": np.asarray(standard_errors, dtype=np.float64),
        "active": np.asarray(active, dtype=np.bool_),
        "episode_seed": np.full(count, int(episode_seed), dtype=np.int64),
        "macro_index": np.asarray(macro_indices, dtype=np.int64),
        "realization_id": np.full(count, int(realization_id), dtype=np.int64),
    }
    if not all(np.isfinite(value).all() for value in rows.values() if value.dtype != np.bool_):
        raise FloatingPointError("V16-A replay row contains NaN/Inf.")
    return rows


def _concat_rows(row_groups: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not row_groups:
        raise ValueError("V16-A replay cannot be empty.")
    keys = tuple(row_groups[0])
    if any(tuple(group) != keys for group in row_groups):
        raise ValueError("V16-A replay schemas differ across episodes.")
    return {key: np.concatenate([group[key] for group in row_groups], axis=0) for key in keys}


def _target_arrays(replay: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    active = np.asarray(replay["active"], dtype=bool)
    active_delta = np.asarray(replay["raw_delta"], dtype=np.float64)[active]
    if not len(active_delta):
        raise ValueError("V16-A cumulative replay contains no active teacher samples.")
    scale = max(_mad(active_delta), 1.0e-8)
    targets = np.zeros_like(np.asarray(replay["raw_delta"], dtype=np.float64))
    targets[active] = np.clip(active_delta / scale, -5.0, 5.0)
    if not np.isfinite(targets).all():
        raise FloatingPointError("V16-A continuous target contains NaN/Inf.")
    if np.any((active_delta > 0.0) & (targets[active] <= 0.0)) or np.any(
        (active_delta < 0.0) & (targets[active] >= 0.0)
    ):
        raise AssertionError("V16-A target sign check failed on cumulative replay.")
    return targets.astype(np.float32), float(scale)


def _to_device(array: np.ndarray, device: torch.device, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def _evaluate_policy(
    actor: Any,
    replay: dict[str, np.ndarray],
    targets: np.ndarray,
) -> dict[str, Any]:
    device = next(actor.parameters()).device
    observations = _to_device(
        replay["normalized_trigger_observation"], device, dtype=torch.float32
    )
    with torch.no_grad():
        probabilities_t = _trigger_probabilities(actor, observations)
    probabilities = probabilities_t.detach().cpu().numpy().astype(np.float64)
    active = np.asarray(replay["active"], dtype=bool)
    deltas = np.asarray(replay["raw_delta"], dtype=np.float64)
    active_spearman = _spearman(probabilities[active], deltas[active])
    active_losses = F.smooth_l1_loss(
        _trigger_logit(actor, _to_device(
            replay["normalized_trigger_observation"][active], device, dtype=torch.float32
        )),
        _to_device(targets[active], device, dtype=torch.float32),
        beta=DELTA_HUBER_BETA,
        reduction="none",
    )
    active_loss_equal = float(
        training._equal_realization_tensor_mean(
            active_losses, replay["realization_id"][active]
        ).detach().item()
    )
    entropy = -probabilities * np.log(np.clip(probabilities, 1.0e-8, 1.0)) - (
        1.0 - probabilities
    ) * np.log(np.clip(1.0 - probabilities, 1.0e-8, 1.0))
    if not np.isfinite(probabilities).all() or not np.isfinite(entropy).all():
        raise FloatingPointError("V16-A policy evaluation contains NaN/Inf.")
    return {
        "p_update_mean": float(np.mean(probabilities)),
        "p_update_state_std": float(np.std(probabilities)),
        "trigger_entropy": float(np.mean(entropy)),
        "train_replay_spearman": active_spearman,
        "train_replay_huber_loss": active_loss_equal,
        "active_probabilities": probabilities,
    }


def _fresh_diagnostics(
    policy: Any,
    fresh_groups: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    actor = policy.macro_cache_trigger_actor
    all_fresh = _concat_rows(fresh_groups)
    device = next(actor.parameters()).device
    observations = _to_device(
        all_fresh["normalized_trigger_observation"], device, dtype=torch.float32
    )
    with torch.no_grad():
        p_all = _trigger_probabilities(actor, observations).detach().cpu().numpy().astype(np.float64)
    per_realization: list[float | None] = []
    per_realization_counts: list[int] = []
    for group in fresh_groups:
        active = np.asarray(group["active"], dtype=bool)
        p = _trigger_probabilities(
            actor,
            _to_device(group["normalized_trigger_observation"], device, dtype=torch.float32),
        ).detach().cpu().numpy().astype(np.float64)
        per_realization.append(_spearman(p[active], group["raw_delta"][active]))
        per_realization_counts.append(int(active.sum()))
    active = np.asarray(all_fresh["active"], dtype=bool)
    values = [float(value) for value in per_realization if value is not None]
    result = {
        "p_update_mean": float(np.mean(p_all)),
        "p_update_std": float(np.std(p_all)),
        "pooled_policy_delta_spearman": _spearman(
            p_all[active], all_fresh["raw_delta"][active]
        ),
        "realization_fresh_preupdate_spearman": per_realization,
        "realization_active_counts": per_realization_counts,
        "fresh_preupdate_spearman_mean": float(np.mean(values)) if values else None,
        "fresh_preupdate_spearman_min": float(np.min(values)) if values else None,
        "fresh_preupdate_spearman_std": float(np.std(values)) if values else None,
        "active_samples": int(active.sum()),
        "macro_states": int(len(all_fresh["active"])),
    }
    if not _finite(result):
        raise FloatingPointError("V16-A fresh diagnostics contain NaN/Inf.")
    return result


def _collection_teacher_summary(collections: list[Any]) -> dict[str, Any]:
    deltas: list[float] = []
    active_deltas: list[float] = []
    standard_errors: list[float] = []
    active_count = 0
    for collection in collections:
        for item in collection.macro.transitions:
            if not (item.trigger_learning_valid and item.trigger_teacher_valid):
                continue
            delta = float(item.teacher_delta_update_keep)
            se = float(item.trigger_update_standard_error)
            deltas.append(delta)
            standard_errors.append(se)
            if item.trigger_credit_active:
                active_count += 1
                active_deltas.append(delta)
    delta = np.asarray(deltas, dtype=np.float64)
    active_delta = np.asarray(active_deltas, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    eligible_count = len(delta)
    return {
        "delta_positive_fraction": float(np.mean(delta > 0.0)) if eligible_count else None,
        "delta_negative_fraction": float(np.mean(delta < 0.0)) if eligible_count else None,
        "active_fraction": float(active_count / eligible_count) if eligible_count else None,
        "raw_delta_median": float(np.median(delta)) if eligible_count else None,
        "raw_delta_mad": _mad(delta) if eligible_count else None,
        "active_raw_delta_median": float(np.median(active_delta)) if len(active_delta) else None,
        "active_raw_delta_mad": _mad(active_delta) if len(active_delta) else None,
        "se_update_mean": float(np.mean(se)) if len(se) else None,
        "se_update_median": float(np.median(se)) if len(se) else None,
        "se_update_p90": float(np.quantile(se, 0.9)) if len(se) else None,
        "teacher_eligible_samples": int(eligible_count),
        "active_samples": int(active_count),
    }


def _save_checkpoint(
    trainer: Any,
    checkpoint_path: Path,
    *,
    raw_hash: str,
    semantic_hash: str,
    episode_pool: Path,
    seed: int,
    completed: int,
    updates: int,
    replay_path: Path,
    report_path: Path,
    best_return: float | None,
) -> None:
    payload = trainer.checkpoint_payload(
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        episode_pool=str(episode_pool),
        seed=seed,
        best_episode_return=best_return,
    )
    payload["learning_method"] = "direct_trigger_delta_distillation"
    payload["v16a_protocol"] = {
        "trigger_ppo_active": False,
        "continuous_target": "clip(raw_DELTA / MAD(active cumulative replay), -5, 5)",
        "loss": "realization_equal_weighted_SmoothL1_beta_1",
        "max_epochs_per_update": DISTILLATION_MAX_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": DISTILLATION_LR,
        "max_grad_norm": DISTILLATION_MAX_GRAD_NORM,
        "policy_kl_hard_limit": DISTILLATION_KL_LIMIT,
        "replay_dataset": str(replay_path.resolve()),
        "report": str(report_path.resolve()),
        "completed_episodes": int(completed),
        "synchronized_updates_completed": int(updates),
        "realizations_per_update": REALIZATIONS_PER_UPDATE,
        "training_episode_seeds": list(range(20001, 20001 + completed)),
    }
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _final_report(
    *,
    update_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    replay: dict[str, np.ndarray],
    trainer: Any,
    checks: dict[str, Any],
    normalization_source: dict[str, Any],
    completed: int,
    rollback_count: int,
    accepted_kls: list[float],
    normalization_stable: bool,
    all_finite: bool,
    formal_runner_path: Path,
    formal_command: str,
    run_dir: Path,
) -> dict[str, Any]:
    targets, delta_scale = _target_arrays(replay)
    final_evaluation = _evaluate_policy(
        trainer.policy.macro_cache_trigger_actor, replay, targets
    )
    update_fresh = {
        int(record["synchronized_update"]): record.get("fresh_preupdate_policy_delta_spearman")
        for record in update_records
    }
    fresh_2_5 = [update_fresh.get(index) for index in range(2, 6)]
    fresh_3_5 = [update_fresh.get(index) for index in range(3, 6)]
    realization_spearmans = [
        float(value)
        for record in update_records
        if 3 <= int(record["synchronized_update"]) <= 5
        for value in record.get("realization_fresh_preupdate_spearman", [])
        if value is not None
    ]
    qualified_fresh_updates = sum(
        value is not None and float(value) >= 0.20 for value in fresh_3_5
    )
    p_final = float(final_evaluation["p_update_mean"])
    state_std = float(final_evaluation["p_update_state_std"])
    collapsed = not (0.05 < p_final < 0.95)
    fresh_3_5_mean = _mean(fresh_3_5)
    realization_mean = float(np.mean(realization_spearmans)) if realization_spearmans else None
    direct_supported = bool(
        not collapsed
        and state_std > 0.02
        and fresh_3_5_mean is not None
        and fresh_3_5_mean >= 0.25
        and qualified_fresh_updates >= 2
        and realization_mean is not None
        and realization_mean > 0.0
        and normalization_stable
        and all_finite
    )
    if direct_supported:
        failure_mode = "NONE"
    elif (
        final_evaluation["train_replay_spearman"] is not None
        and float(final_evaluation["train_replay_spearman"]) >= 0.25
        and (fresh_3_5_mean is None or fresh_3_5_mean < 0.20 or qualified_fresh_updates < 2)
    ):
        failure_mode = "STATE_REPRESENTATION_GENERALIZATION_LIMIT"
    elif collapsed:
        failure_mode = "TRIGGER_COLLAPSE"
    elif not normalization_stable:
        failure_mode = "NORMALIZATION_UNSTABLE"
    elif not all_finite:
        failure_mode = "NONFINITE_VALUE"
    else:
        failure_mode = "FRESH_PREUPDATE_ALIGNMENT_BELOW_GATE"
    non_trigger_steps = sum(
        int(trainer.training_counters.get(key, 0))
        for key in (
            "micro_optimizer_steps",
            "macro_optimizer_steps",
            "micro_actor_optimizer_steps",
            "micro_critic_optimizer_steps",
            "macro_actor_optimizer_steps",
            "macro_critic_optimizer_steps",
            "long_value_optimizer_steps",
        )
    )
    return {
        "status": "COMPLETED",
        "experiment": "V16-A direct Trigger DELTA distillation gate",
        "formal_runner_path": str(formal_runner_path.resolve()),
        "formal_command": formal_command,
        "run_dir": str(run_dir.resolve()),
        "IMPLEMENTATION_COMPLETE": True,
        "TRIGGER_PPO_ACTIVE": False,
        "DIRECT_TRIGGER_DISTILLATION_ACTIVE": True,
        "EPISODES_COMPLETED": int(completed),
        "SYNC_UPDATES_COMPLETED": len(update_records),
        "REALIZATIONS_PER_UPDATE": REALIZATIONS_PER_UPDATE,
        "NON_TRIGGER_OPTIMIZER_STEPS": int(non_trigger_steps),
        "REPLAY_REALIZATIONS_FINAL": int(len(np.unique(replay["realization_id"]))),
        "REPLAY_ACTIVE_SAMPLES_FINAL": int(np.count_nonzero(replay["active"])),
        "DELTA_SCALE_FINAL": delta_scale,
        "FRESH_PREUPDATE_SPEARMAN_UPDATE2": update_fresh.get(2),
        "FRESH_PREUPDATE_SPEARMAN_UPDATE3": update_fresh.get(3),
        "FRESH_PREUPDATE_SPEARMAN_UPDATE4": update_fresh.get(4),
        "FRESH_PREUPDATE_SPEARMAN_UPDATE5": update_fresh.get(5),
        "FRESH_PREUPDATE_SPEARMAN_UPDATES_2_5_MEAN": _mean(fresh_2_5),
        "FRESH_PREUPDATE_SPEARMAN_UPDATES_3_5_MEAN": fresh_3_5_mean,
        "FRESH_PREUPDATE_REALIZATION_MEAN_UPDATES_3_5": realization_mean,
        "FRESH_PREUPDATE_REALIZATION_MIN_UPDATES_3_5": (
            float(np.min(realization_spearmans)) if realization_spearmans else None
        ),
        "TRAIN_REPLAY_SPEARMAN_FINAL": final_evaluation["train_replay_spearman"],
        "TRAIN_REPLAY_HUBER_LOSS_FINAL": final_evaluation["train_replay_huber_loss"],
        "P_UPDATE_FINAL": p_final,
        "P_UPDATE_STATE_STD_FINAL": state_std,
        "TRIGGER_ENTROPY_FINAL": final_evaluation["trigger_entropy"],
        "RETURN_FIRST_UPDATE_MEAN": (
            update_records[0].get("return_mean_of_realizations") if update_records else None
        ),
        "RETURN_LAST_UPDATE_MEAN": (
            update_records[-1].get("return_mean_of_realizations") if update_records else None
        ),
        "MAX_ACCEPTED_POLICY_KL": max(accepted_kls) if accepted_kls else 0.0,
        "DISTILLATION_KL_ROLLBACK_COUNT": int(rollback_count),
        "NORMALIZATION_STABLE": bool(normalization_stable),
        "TRIGGER_COLLAPSE": bool(collapsed),
        "DIRECT_DISTILLATION_GENERALIZATION_SUPPORTED": direct_supported,
        "TRIGGER_LEARNING_REPAIR_SUPPORTED": direct_supported,
        "READY_FOR_STAGE2": direct_supported,
        "PRIMARY_FAILURE_MODE": failure_mode,
        "ARCHITECTURE_VERSION": int(trainer.policy.architecture_version),
        "CHECKPOINT_VERSION": int(training.CHECKPOINT_VERSION),
        "NON_TRIGGER_MODULES_UNCHANGED": True,
        "TRIGGER_PARAMS_FROZEN_DURING_COLLECTION": True,
        "REALIZATION_EQUAL_WEIGHTING": True,
        "NORMALIZATION_SOURCE": normalization_source,
        "MINIMAL_CHECKS": checks,
        "SYNC_UPDATE_METRICS": update_records,
        "EPISODE_METRICS": episode_records,
        "TRAINING_COUNTERS": dict(trainer.training_counters),
    }


def run_v16a_direct_trigger_distillation(
    *,
    config_path: Path,
    episode_pool: Path,
    run_dir: Path,
    report_path: Path,
    episode_seed_start: int,
    base_seed: int,
    initial_checkpoint_path: Path,
    initial_checkpoint_sha256: str,
    normalization: dict[str, Any],
    normalization_source: dict[str, Any],
    formal_runner_path: Path,
    formal_command: str,
) -> dict[str, Any]:
    checks = _minimal_checks()
    cfg = training.project_config.load_config(str(config_path.resolve(strict=True)))
    training.project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    realization_paths = training._resolve_training_episodes(
        project_root=project_root,
        config_path=config_path,
        episode_pool=episode_pool,
        episode_seed_start=episode_seed_start,
        count=TARGET_EPISODES,
        generate_missing=False,
        generation_device="cpu",
    )
    if len(realization_paths) != TARGET_EPISODES:
        raise AssertionError("V16-A requires exactly twenty existing training realizations.")
    raw_hash = training.project_config.raw_config_sha256(config_path)
    first_context = load_environment_context(
        str(config_path), realization_dir=realization_paths[0]
    )
    semantic_hash = training.project_config.semantic_config_hash(first_context.cfg)
    pool_path = training._resolve_path(project_root, episode_pool)
    run_path = run_dir.resolve(strict=False)
    if run_path.parent != (project_root / "runs").resolve(strict=False):
        raise ValueError("V16-A run directory must be a direct child of runs/.")
    training.ensure_fresh_run_dir_is_clean(run_path)

    training._seed_everything(int(base_seed))
    trainer = training.PPOTrainer(
        first_context,
        macro_actor_learning_rate_override=DISTILLATION_LR,
        stage1_trigger_only=True,
        trigger_observation_normalization=normalization,
    )
    warmstart_metadata = trainer.load_v14b_downstream_checkpoint(
        initial_checkpoint_path,
        expected_sha256=initial_checkpoint_sha256,
    )
    trainer.initialization_metadata["trigger_learning_protocol"] = "direct_delta_distillation"
    actor = trainer.policy.macro_cache_trigger_actor
    trigger_optimizer = trainer.macro_cache_trigger_actor_optimizer
    for group in trigger_optimizer.param_groups:
        group["lr"] = DISTILLATION_LR
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    for parameter in trainer.policy.macro_cache_trigger_actor_parameters():
        parameter.requires_grad_(True)
    trigger_parameter_ids = {id(parameter) for parameter in actor.parameters()}
    if any(
        parameter.requires_grad and id(parameter) not in trigger_parameter_ids
        for parameter in trainer.policy.parameters()
    ):
        raise AssertionError("V16-A unexpectedly left a non-Trigger parameter trainable.")
    if not trainer.policy.trigger_observation_normalization_active:
        raise AssertionError("V16-A must inherit the verified C2/C4 normalizer.")
    normalization_probe = trainer.policy._normalized_trigger_observations(
        torch.zeros((2, 400), dtype=torch.float32, device=trainer.device)
    )
    normalization_stable = bool(
        torch.isfinite(normalization_probe).all().item()
        and float(normalization_probe.abs().max().item()) <= 10.0
        and int(normalization_source.get("low_variance_feature_count", -1)) == 73
    )
    if not normalization_stable:
        raise FloatingPointError("V16-A normalization finite/clip/source check failed.")
    checks["normalization_finite"] = {"pass": True, "feature_count": 400, "postclip_abs_max": 10.0}

    run_path.mkdir(parents=True, exist_ok=True)
    run_config_path = run_path / "run_config.toml"
    shutil.copyfile(config_path, run_config_path)
    training.ensure_run_artifact_directories(run_path)
    manifest_path = run_path / training.RUN_MANIFEST_FILENAME
    training.write_dual_timescale_run_manifest(
        manifest_path,
        config_path=config_path,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        seed=base_seed,
        trainer=trainer,
        context=first_context,
        target_episodes=TARGET_EPISODES,
        episode_pool=str(pool_path),
        episode_seed_start=episode_seed_start,
        resume=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment_id": V16A_RUN_ID,
            "formal_runner_path": str(formal_runner_path.resolve()),
            "formal_command": formal_command,
            "trigger_ppo_active": False,
            "direct_trigger_distillation_active": True,
            "trigger_learning_rule": "continuous_DELTA_MAD_SmoothL1",
            "trigger_optimizer": "Adam",
            "trigger_learning_rate": DISTILLATION_LR,
            "trigger_max_grad_norm": DISTILLATION_MAX_GRAD_NORM,
            "trigger_distillation_max_epochs_per_update": DISTILLATION_MAX_EPOCHS,
            "trigger_policy_kl_hard_limit": DISTILLATION_KL_LIMIT,
            "trigger_replay_cumulative": True,
            "realization_equal_weighting": True,
            "synchronized_realizations_per_update": REALIZATIONS_PER_UPDATE,
            "synchronized_updates_target": SYNC_UPDATES,
            "training_episode_seeds": list(range(episode_seed_start, episode_seed_start + TARGET_EPISODES)),
            "normalization_source_sha256": normalization_source["source_sha256"],
            "source_v14b_warmstart": warmstart_metadata,
        }
    )
    _write_json(manifest_path, manifest)

    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    rollouts_dir = run_path / "rollouts"
    replay_path = run_path / "trigger_replay_dataset.npz"
    checkpoint_path = run_path / "checkpoints" / training.CHECKPOINT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    initial_report = {
        "status": "IN_PROGRESS",
        "experiment": "V16-A direct Trigger DELTA distillation gate",
        "run_dir": str(run_path.resolve()),
        "formal_runner_path": str(formal_runner_path.resolve()),
        "formal_command": formal_command,
        "target_episodes": TARGET_EPISODES,
        "target_synchronized_updates": SYNC_UPDATES,
        "realizations_per_update": REALIZATIONS_PER_UPDATE,
        "training_episode_seeds": list(range(episode_seed_start, episode_seed_start + TARGET_EPISODES)),
        "normalization_source": normalization_source,
        "minimal_checks": checks,
        "completed_episodes": 0,
        "completed_updates": 0,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "replay_dataset_path": str(replay_path.resolve()),
    }
    _write_json(report_path, initial_report)

    trigger_state_before = _module_sha256(actor)
    nontrigger_state_before = _module_sha256(trainer.policy, exclude_trigger=True)
    replay_groups: list[dict[str, np.ndarray]] = []
    update_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    accepted_kls: list[float] = []
    rollback_count = 0
    best_return: float | None = None
    completed = 0

    for update_index in range(SYNC_UPDATES):
        update_started = time.perf_counter()
        policy_hash = _module_sha256(actor)
        policy_snapshot = {
            key: value.detach().clone() for key, value in actor.state_dict().items()
        }
        nontrigger_hash = _module_sha256(trainer.policy, exclude_trigger=True)
        optimizer_steps_before = int(
            trainer.training_counters.get("macro_cache_trigger_optimizer_steps", 0)
        )
        new_groups: list[dict[str, np.ndarray]] = []
        new_collections: list[Any] = []
        seeds = [episode_seed_start + update_index * REALIZATIONS_PER_UPDATE + offset for offset in range(REALIZATIONS_PER_UPDATE)]

        for offset, episode_seed in enumerate(seeds):
            episode_index = completed + offset
            realization_path = realization_paths[episode_index]
            context = load_environment_context(str(config_path), realization_dir=realization_path)
            training._assert_realization_context_compatible(first_context, context)
            trainer.context = context
            collection = collect_dual_timescale_episode(
                trainer.policy,
                context,
                training.aligned_train_steps(context),
                trainer.duration_candidates,
                deterministic=False,
                critic_independent_trigger_teacher=True,
                base_seed=int(base_seed),
                episode_seed=int(episode_seed),
                rollout_index=episode_index + 1,
            )
            rows = _collection_rows(
                collection,
                policy=trainer.policy,
                episode_seed=episode_seed,
                realization_id=episode_seed,
            )
            _write_npz_atomic(
                rollouts_dir / f"episode_{episode_seed}_trigger_transitions.npz", rows
            )
            current_state = actor.state_dict()
            if _module_sha256(actor) != policy_hash or any(
                not torch.equal(policy_snapshot[key], current_state[key])
                for key in policy_snapshot
            ):
                raise RuntimeError("Trigger changed during the frozen four-realization collection.")
            if _module_sha256(trainer.policy, exclude_trigger=True) != nontrigger_hash:
                raise RuntimeError("A non-Trigger policy module changed during collection.")
            if int(trainer.training_counters.get("macro_cache_trigger_optimizer_steps", 0)) != optimizer_steps_before:
                raise RuntimeError("Trigger optimizer stepped before the four-realization group completed.")
            if not _finite(collection.summary):
                raise FloatingPointError("V16-A rollout summary contains NaN/Inf.")
            if not bool(collection.summary.get("counterfactual_state_restore_valid")):
                raise RuntimeError("V16-A teacher counterfactual state restore failed.")
            if not bool(collection.summary.get("rollout_rng_unchanged")):
                raise RuntimeError("V16-A teacher counterfactual RNG restore failed.")
            new_groups.append(rows)
            new_collections.append(collection)

        if _module_sha256(actor) != policy_hash or _module_sha256(trainer.policy, exclude_trigger=True) != nontrigger_hash:
            raise RuntimeError("V16-A four-realization frozen policy guard failed at group end.")

        fresh_preupdate = _fresh_diagnostics(trainer.policy, new_groups)
        collection_summary = _collection_teacher_summary(new_collections)
        replay_groups.extend(new_groups)
        replay = _concat_rows(replay_groups)
        replay_realizations = len(np.unique(replay["realization_id"]))
        if replay_realizations != (update_index + 1) * REALIZATIONS_PER_UPDATE:
            raise AssertionError("V16-A cumulative replay did not grow by four realizations.")
        checks["cumulative_replay_growth"]["observed_realization_counts"] = [
            int(record.get("replay_realization_count", 0)) for record in update_records
        ] + [replay_realizations]
        checks["four_realization_collection_freeze"] = {
            "pass": True,
            "updates_verified": update_index + 1,
            "realizations_per_update": REALIZATIONS_PER_UPDATE,
            "optimizer_steps_during_collection": 0,
        }
        targets, delta_scale = _target_arrays(replay)
        active_mask = np.asarray(replay["active"], dtype=bool)
        device = trainer.device
        replay_observations = _to_device(
            replay["normalized_trigger_observation"], device, dtype=torch.float32
        )
        active_observations = _to_device(
            replay["normalized_trigger_observation"][active_mask], device, dtype=torch.float32
        )
        active_targets = _to_device(targets[active_mask], device, dtype=torch.float32)
        active_realization_ids = replay["realization_id"][active_mask]
        training_stats: list[dict[str, Any]] = []
        accepted_update_kls: list[float] = []
        rejected_this_update = 0
        epoch_attempts = 0
        for _epoch in range(DISTILLATION_MAX_EPOCHS):
            epoch_attempts += 1
            epoch_result = _transactional_epoch_step(
                actor,
                trigger_optimizer,
                replay_observations=replay_observations,
                active_observations=active_observations,
                active_targets=active_targets,
                active_realization_ids=active_realization_ids,
                hard_kl=DISTILLATION_KL_LIMIT,
            )
            training_stats.append(epoch_result)
            if epoch_result["rolled_back"]:
                rejected_this_update += 1
                rollback_count += 1
                break
            if epoch_result["accepted"]:
                accepted_update_kls.append(float(epoch_result["mean_policy_kl"]))
                accepted_kls.append(float(epoch_result["mean_policy_kl"]))
                trainer.training_counters["macro_cache_trigger_optimizer_steps"] = int(
                    trainer.training_counters.get("macro_cache_trigger_optimizer_steps", 0)
                ) + 1
            if not _module_is_finite(actor):
                raise FloatingPointError("V16-A Trigger parameters contain NaN/Inf.")
        trainer.training_counters["completed_episodes"] = completed + REALIZATIONS_PER_UPDATE
        trainer.training_counters["synchronized_trigger_updates"] = update_index + 1

        post_fresh = _fresh_diagnostics(trainer.policy, new_groups)
        final_target_eval = _evaluate_policy(actor, replay, targets)
        episode_returns: list[float] = []
        cache_rates: list[float] = []
        for offset, (episode_seed, collection, rows) in enumerate(
            zip(seeds, new_collections, new_groups, strict=True)
        ):
            summary = collection.summary
            episode_return = float(summary["episode_return"])
            episode_returns.append(episode_return)
            actual_update_rate = float(summary["actual_cache_update_rate"])
            cache_rates.append(actual_update_rate)
            if best_return is None or episode_return > best_return:
                best_return = episode_return
            record = {
                "record_type": "episode",
                "episode": int(completed + offset + 1),
                "episode_seed": int(episode_seed),
                "realization_id": int(episode_seed),
                "synchronized_update": int(update_index + 1),
                "realization_in_update": int(offset + 1),
                "macro_state_count": int(len(rows["macro_index"])),
                "active_sample_count": int(np.count_nonzero(rows["active"])),
                "episode_return": episode_return,
                "actual_cache_update_rate": actual_update_rate,
                "requested_duration_mean": summary.get("requested_duration_mean"),
                "trigger_teacher_valid_count": summary.get("trigger_teacher_valid_count"),
                "counterfactual_branch_count": summary.get("counterfactual_branch_count"),
                "episode_runtime_seconds": summary.get("episode_runtime_seconds"),
                "trigger_policy_version_during_collection": policy_hash,
            }
            if not _finite(record):
                raise FloatingPointError("V16-A episode record contains NaN/Inf.")
            _append_jsonl(episode_metrics_path, record)
            episode_records.append(record)

        completed += REALIZATIONS_PER_UPDATE
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        _write_npz_atomic(replay_path, replay)
        update_record = {
            "record_type": "update",
            "synchronized_update": update_index + 1,
            "episode_seed_start": seeds[0],
            "episode_seed_end": seeds[-1],
            "realization_seeds": seeds,
            "old_trigger_policy_version": policy_hash,
            "trigger_params_bitwise_frozen_during_collection": True,
            "nontrigger_modules_unchanged_during_collection": True,
            "fresh_preupdate_p_update_mean": fresh_preupdate["p_update_mean"],
            "fresh_preupdate_p_update_std": fresh_preupdate["p_update_std"],
            "fresh_preupdate_policy_delta_spearman": fresh_preupdate["pooled_policy_delta_spearman"],
            "realization_fresh_preupdate_spearman": fresh_preupdate["realization_fresh_preupdate_spearman"],
            "fresh_preupdate_spearman_mean": fresh_preupdate["fresh_preupdate_spearman_mean"],
            "fresh_preupdate_spearman_min": fresh_preupdate["fresh_preupdate_spearman_min"],
            "fresh_preupdate_spearman_std": fresh_preupdate["fresh_preupdate_spearman_std"],
            "fresh_preupdate_macro_states": fresh_preupdate["macro_states"],
            "fresh_preupdate_active_samples": fresh_preupdate["active_samples"],
            **collection_summary,
            "replay_realization_count": replay_realizations,
            "replay_state_count": int(len(replay["active"])),
            "replay_active_sample_count": int(np.count_nonzero(replay["active"])),
            "delta_scale_mad_active_cumulative_replay": delta_scale,
            "train_replay_spearman": final_target_eval["train_replay_spearman"],
            "train_replay_huber_loss": final_target_eval["train_replay_huber_loss"],
            "supervised_epochs_attempted": epoch_attempts,
            "supervised_epochs_accepted": sum(bool(row["accepted"]) for row in training_stats),
            "supervised_grad_norm_by_epoch": [row["grad_norm"] for row in training_stats],
            "accepted_policy_kl_by_epoch": accepted_update_kls,
            "accepted_policy_kl_max": max(accepted_update_kls) if accepted_update_kls else 0.0,
            "distillation_kl_rollback_count": rejected_this_update,
            "postupdate_p_update_mean": post_fresh["p_update_mean"],
            "postupdate_p_update_std": post_fresh["p_update_std"],
            "postupdate_policy_delta_spearman": post_fresh["pooled_policy_delta_spearman"],
            "postupdate_realization_spearman": post_fresh["realization_fresh_preupdate_spearman"],
            "return_mean_of_realizations": float(np.mean(episode_returns)),
            "actual_cache_update_rate_mean": float(np.mean(cache_rates)),
            "synchronized_collection_runtime_seconds": float(time.perf_counter() - update_started),
            "trigger_supervised_optimizer_steps_this_update": sum(bool(row["accepted"]) for row in training_stats),
            "non_trigger_optimizer_steps_this_update": 0,
        }
        if not _finite(update_record):
            raise FloatingPointError("V16-A update record contains NaN/Inf.")
        _append_jsonl(update_metrics_path, update_record)
        update_records.append(update_record)
        _save_checkpoint(
            trainer,
            checkpoint_path,
            raw_hash=raw_hash,
            semantic_hash=semantic_hash,
            episode_pool=pool_path,
            seed=base_seed,
            completed=completed,
            updates=update_index + 1,
            replay_path=replay_path,
            report_path=report_path,
            best_return=best_return,
        )
        manifest.update(
            {
                "completed_episodes": completed,
                "synchronized_updates_completed": update_index + 1,
                "replay_realization_count": replay_realizations,
                "replay_active_sample_count": int(np.count_nonzero(replay["active"])),
                "latest_checkpoint": str(checkpoint_path.resolve()),
                "trigger_replay_dataset": str(replay_path.resolve()),
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(
            run_path / training.RUN_SUMMARY_FILENAME,
            {
                "experiment_type": V16A_RUN_ID,
                "architecture_version": int(trainer.policy.architecture_version),
                "checkpoint_version": int(training.CHECKPOINT_VERSION),
                "completed_episodes": completed,
                "target_episodes": TARGET_EPISODES,
                "synchronized_updates_completed": update_index + 1,
                "realizations_per_update": REALIZATIONS_PER_UPDATE,
                "stage1_trigger_only": True,
                "trigger_ppo_active": False,
                "direct_trigger_distillation_active": True,
                "trigger_supervised_optimizer_steps": int(
                    trainer.training_counters.get("macro_cache_trigger_optimizer_steps", 0)
                ),
                "non_trigger_optimizer_steps": 0,
                "best_episode_return": best_return,
                "last_update": update_record,
                "initialization_metadata": trainer.initialization_metadata,
            },
        )
        progress = {
            **initial_report,
            "status": "IN_PROGRESS",
            "completed_episodes": completed,
            "completed_updates": update_index + 1,
            "SYNC_UPDATE_METRICS": update_records,
            "EPISODE_METRICS": episode_records,
            "replay_realizations": replay_realizations,
            "replay_active_samples": int(np.count_nonzero(replay["active"])),
            "latest_checkpoint": str(checkpoint_path.resolve()),
        }
        _write_json(report_path, progress)
        print(
            json.dumps(
                {
                    "synchronized_update": update_index + 1,
                    "episodes_completed": completed,
                    "seeds": seeds,
                    "fresh_preupdate_spearman": fresh_preupdate["pooled_policy_delta_spearman"],
                    "realization_fresh_preupdate_spearman": fresh_preupdate[
                        "realization_fresh_preupdate_spearman"
                    ],
                    "replay_realizations": replay_realizations,
                    "replay_active_samples": int(np.count_nonzero(replay["active"])),
                    "accepted_kl_max": update_record["accepted_policy_kl_max"],
                    "rollbacks": rejected_this_update,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    if _module_sha256(trainer.policy, exclude_trigger=True) != nontrigger_state_before:
        raise RuntimeError("V16-A altered a frozen non-Trigger module.")
    if not all(_finite(record) for record in update_records + episode_records):
        raise FloatingPointError("V16-A final artifacts contain NaN/Inf.")
    checks["no_nan_inf"] = {"pass": True, "checked_replay_policy_metrics_and_episode_records": True}
    report = _final_report(
        update_records=update_records,
        episode_records=episode_records,
        replay=replay,
        trainer=trainer,
        checks=checks,
        normalization_source=normalization_source,
        completed=completed,
        rollback_count=rollback_count,
        accepted_kls=accepted_kls,
        normalization_stable=normalization_stable,
        all_finite=True,
        formal_runner_path=formal_runner_path,
        formal_command=formal_command,
        run_dir=run_path,
    )
    report["replay_dataset_path"] = str(replay_path.resolve())
    report["checkpoint_path"] = str(checkpoint_path.resolve())
    report["best_episode_return"] = best_return
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "IMPLEMENTATION_COMPLETE",
                    "EPISODES_COMPLETED",
                    "SYNC_UPDATES_COMPLETED",
                    "FRESH_PREUPDATE_SPEARMAN_UPDATES_3_5_MEAN",
                    "TRAIN_REPLAY_SPEARMAN_FINAL",
                    "TRIGGER_COLLAPSE",
                    "DIRECT_DISTILLATION_GENERALIZATION_SUPPORTED",
                    "READY_FOR_STAGE2",
                    "PRIMARY_FAILURE_MODE",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return report
