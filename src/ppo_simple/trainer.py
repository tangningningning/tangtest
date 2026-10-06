"""训练 Macro Cache SMDP PPO 与 Micro UAV autoregressive PPO。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 仅静态导入环境
    torch = None  # type: ignore[assignment]

from common import project_config
from scenario.realization import (
    REALIZATION_ARTIFACT_FILENAMES,
    _realization_artifacts_from_directory,
)
from .interface import (
    PROPOSED_POLICY_SCHEMA,
    _assert_realization_context_compatible,
    aligned_train_steps,
    load_environment_context,
    macro_duration_candidates_from_project,
)
from .macro_targets import (
    build_macro_n2_target,
)
from .policy import DualTimescaleActorCritic
from .primitive_control import PrimitiveTensorSpec
from .reporting import (
    RUN_MANIFEST_FILENAME,
    RUN_SUMMARY_FILENAME,
    SLOT_METRICS_FILENAME,
    append_dual_timescale_record,
    append_dual_timescale_slot_metrics,
    build_macro_transition_trace_arrays,
    checkpoint_path_for_resume,
    checkpoint_path_for_write,
    ensure_fresh_run_dir_is_clean,
    ensure_run_artifact_directories,
    format_training_epoch_log,
    update_training_epoch_return_plot,
    write_dual_timescale_run_manifest,
    write_macro_transition_trace_artifact,
)
from .rollout import (
    CounterfactualRestoreError,
    EpisodeCollection,
    MacroRolloutBuffer,
    MicroRolloutBuffer,
    collect_dual_timescale_episode,
    compute_gae,
    compute_macro_smdp_gae,
)

CHECKPOINT_METHOD = "ppo_dual_timescale_v15c2_trigger_normalization"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 47
UNIFIED_CACHE_CHECKPOINT_VERSION = 48
UNIFIED_CACHE_CARDINALITY_CHECKPOINT_VERSION = 49
UNIFIED_CACHE_ALGORITHM_ID = "UNIFIED_CACHE_ACTION_V1"
UNIFIED_CACHE_CHECKPOINT_METHOD = "ppo_unified_cache_candidate_smdp_micro_uav_v16"
UNIFIED_CACHE_CARDINALITY_ALGORITHM_ID = "UNIFIED_CACHE_ACTION_CARDINALITY_CORRECTED_V1"
UNIFIED_CACHE_CARDINALITY_CHECKPOINT_METHOD = (
    "ppo_unified_cache_candidate_cardinality_corrected_smdp_micro_uav_v17"
)
PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
ALGORITHM_ID = "V15-C2_TRIGGER_NORMALIZATION"
MACRO_ACTOR_ADVANTAGE_MODE = "v15c2_full_information_transactional_binary_trigger_ppo_unit_scale_clamp10"
TRIGGER_CANDIDATE_SAMPLES = 4
TRIGGER_ADVANTAGE_EPSILON = 1.0e-8
TRIGGER_ADVANTAGE_CLIP = 5.0
TRIGGER_EARLY_GATE_EPISODES = 10
TRIGGER_KL_PERSISTENCE_EPISODES = 2
TRIGGER_COLLAPSE_PERSISTENCE_EPISODES = 3
MACRO_CRITIC_TARGET_MODE = "two_step_smdp_td"
MACRO_CRITIC_LAST1_ALPHA = 1.0
TRIGGER_FULL_INFORMATION_PPO_CLIP = 0.2
TRIGGER_TARGET_KL = 0.02
TRIGGER_HARD_KL = 0.05
TRIGGER_BACKTRACKING_STEP_SCALES = (1.0, 0.25, 0.0625, 0.015625)
TRIGGER_MAX_GRAD_NORM = 0.5
V14B_CHECKPOINT_VERSION = 37
V14B_ARCHITECTURE_VERSION = 15
V14B_LATEST_CHECKPOINT_SHA256 = "82a835d7e5be6325566f14ffa65afeb388d7e3ba47e4ddce487a14e6b7d1e55c"


def _require_torch() -> None:
    if torch is None:
        raise ImportError("正式双时间尺度 PPO 需要 PyTorch。")


def _module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _pool_children(pool_path: Path) -> list[Path]:
    child_directories = [
        child
        for child in pool_path.iterdir()
        if child.is_dir()
        and any((child / filename).is_file() for filename in REALIZATION_ARTIFACT_FILENAMES.values())
    ]
    child_directories.sort(
        key=lambda child: (
            int(match.group(1)) if (match := re.search(r"(\d+)$", child.name)) else math.inf,
            child.name,
        )
    )
    if child_directories:
        return child_directories
    if any((pool_path / filename).is_file() for filename in REALIZATION_ARTIFACT_FILENAMES.values()):
        return [pool_path]
    return []


def _load_episode_generator(project_root: Path) -> Any:
    generator_path = project_root / "scripts" / "generation" / "generate_episode_realization.py"
    if not generator_path.is_file():
        raise FileNotFoundError(f"episode generator missing: {generator_path}")
    module_name = "_adaptive_cache_mec_dual_episode_generator"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load episode generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_training_episodes(
    *,
    project_root: Path,
    config_path: str | Path,
    episode_pool: str | Path | None,
    episode_seed_start: int | None,
    count: int,
    generate_missing: bool,
    generation_device: str,
) -> tuple[Path, ...]:
    if episode_pool is None:
        raise ValueError("双时间尺度训练要求显式指定预生成 --episode-pool。")
    if generate_missing and episode_seed_start is None:
        raise ValueError(
            "--generate-missing-episodes 必须同时指定 --episode-seed-start。"
        )
    pool_path = _resolve_path(project_root, episode_pool)
    if pool_path.exists() and not pool_path.is_dir():
        raise FileNotFoundError(f"episode pool 不是目录：{pool_path}")
    if not pool_path.exists():
        if not generate_missing:
            raise FileNotFoundError(f"episode pool 不存在：{pool_path}")
        pool_path.mkdir(parents=True, exist_ok=False)
    if episode_seed_start is not None:
        seeds = [int(episode_seed_start) + index for index in range(int(count))]
        paths = [pool_path / f"episode_{seed}" for seed in seeds]
        missing = [path for path in paths if not path.is_dir()]
        if missing and not generate_missing:
            raise FileNotFoundError(f"episode realization 缺失：{missing[0]}")
        generator = _load_episode_generator(project_root) if missing else None
        for seed, path in zip(seeds, paths, strict=True):
            if path.exists():
                _realization_artifacts_from_directory(path)
                continue
            path.mkdir(parents=False, exist_ok=False)
            assert generator is not None
            generator.run_episode(
                _resolve_path(project_root, config_path),
                seed,
                path,
                None,
                generation_device,
            )
            _realization_artifacts_from_directory(path)
        return tuple(paths)
    paths = _pool_children(pool_path)
    if not paths:
        raise ValueError(f"episode pool 中没有完整 realization：{pool_path}")
    if int(count) > len(paths):
        raise RuntimeError(
            "episode pool exhausted: "
            f"requested={count}, available={len(paths)}"
        )
    return tuple(paths[: int(count)])


def _normalize_advantages(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(array.mean())
    std = float(array.std())
    if std <= 1.0e-8:
        return np.zeros_like(array), mean, std
    return (array - mean) / (std + 1.0e-8), mean, std


def _clipped_ppo_surrogate_loss(ratio: Any, advantage: Any, clip: float) -> Any:
    return -torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * advantage,
    ).mean()


def _realized_smdp_return_to_go(
    rewards: Any, actual_durations: Any, dones: Any, gamma: float
) -> np.ndarray:
    """Compute an episode's realized SMDP return-to-go for diagnostics only."""

    reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    durations = np.asarray(actual_durations, dtype=np.int64).reshape(-1)
    terminal = np.asarray(dones, dtype=bool).reshape(-1)
    if not (reward_values.shape == durations.shape == terminal.shape):
        raise ValueError("Realized Macro return arrays must have matching shapes.")
    if np.any(durations <= 0) or not np.isfinite(reward_values).all():
        raise ValueError("Realized Macro return inputs are invalid.")
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("Realized Macro return gamma must be finite and non-negative.")
    result = np.zeros_like(reward_values)
    running_return = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        if terminal[index]:
            running_return = 0.0
        running_return = float(
            reward_values[index]
            + gamma_value ** int(durations[index]) * running_return
        )
        result[index] = running_return
    if not np.isfinite(result).all():
        raise ValueError("Realized Macro return-to-go contains NaN/Inf.")
    return result.astype(np.float32)


def _gradient_norm(parameters: Any) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).detach().cpu())


def _pcgrad_lite_current_priority(
    current_gradients: tuple[Any, ...],
    last1_gradients: tuple[Any, ...],
    *,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Match last1's norm, project only it on conflict, then restore current norm."""

    if len(current_gradients) != len(last1_gradients) or not current_gradients:
        raise ValueError("PCGrad groups must contain matching non-empty gradients.")
    current = []
    last1 = []
    for current_gradient, last1_gradient in zip(
        current_gradients, last1_gradients, strict=True
    ):
        if tuple(current_gradient.shape) != tuple(last1_gradient.shape):
            raise ValueError("PCGrad group gradient tensor shapes must match.")
        current.append(current_gradient.detach())
        last1.append(last1_gradient.detach())
    current = tuple(current)
    last1 = tuple(last1)

    def dot(left: tuple[Any, ...], right: tuple[Any, ...]) -> Any:
        return sum(torch.sum(a * b) for a, b in zip(left, right, strict=True))

    def norm(gradients: tuple[Any, ...]) -> Any:
        return torch.sqrt(dot(gradients, gradients))

    current_norm_tensor = norm(current)
    last1_norm_raw_tensor = norm(last1)
    dot_raw_tensor = dot(current, last1)
    current_norm = float(current_norm_tensor.cpu())
    last1_norm_raw = float(last1_norm_raw_tensor.cpu())
    dot_raw = float(dot_raw_tensor.cpu())
    cosine_raw = dot_raw / (current_norm * last1_norm_raw + float(eps))

    current_zero_fallback = current_norm <= float(eps)
    if last1_norm_raw > float(eps) and not current_zero_fallback:
        last_scale_tensor = current_norm_tensor / (last1_norm_raw_tensor + float(eps))
        last1_scaled = tuple(gradient * last_scale_tensor for gradient in last1)
        last_scale = float(last_scale_tensor.cpu())
    else:
        last_scale_tensor = current_norm_tensor.new_zeros(())
        last1_scaled = tuple(torch.zeros_like(gradient) for gradient in last1)
        last_scale = 0.0
    last1_scaled_norm_tensor = norm(last1_scaled)
    last1_scaled_norm = float(last1_scaled_norm_tensor.cpu())
    scaled_to_current_ratio = (
        last1_scaled_norm / current_norm if current_norm > float(eps) else 0.0
    )

    dot_scaled_tensor = dot(current, last1_scaled)
    projection_active = bool(float(dot_scaled_tensor.cpu()) < 0.0)
    if projection_active and not current_zero_fallback:
        projection_coefficient = dot_scaled_tensor / (
            current_norm_tensor.square() + float(eps)
        )
        last1_adjusted = tuple(
            historical - projection_coefficient * present
            for present, historical in zip(current, last1_scaled, strict=True)
        )
    else:
        last1_adjusted = tuple(item.clone() for item in last1_scaled)

    dot_post_tensor = dot(current, last1_adjusted)
    last1_adjusted_norm_tensor = norm(last1_adjusted)
    cosine_post = float(
        dot_post_tensor.cpu()
        / (current_norm_tensor * last1_adjusted_norm_tensor + float(eps))
    )
    mixed = tuple(
        present + MACRO_CRITIC_LAST1_ALPHA * historical
        for present, historical in zip(current, last1_adjusted, strict=True)
    )
    mixed_norm_tensor = norm(mixed)
    mixed_norm = float(mixed_norm_tensor.cpu())
    if current_zero_fallback or mixed_norm <= float(eps):
        combined = tuple(gradient.clone() for gradient in current)
    else:
        final_scale = current_norm_tensor / mixed_norm_tensor
        combined = tuple(gradient * final_scale for gradient in mixed)
    combined_norm_tensor = norm(combined)
    combined_norm = float(combined_norm_tensor.cpu())
    final_to_current_ratio = (
        combined_norm / current_norm if current_norm > float(eps)
        else 1.0 if combined_norm <= float(eps)
        else float("inf")
    )
    if not math.isfinite(final_to_current_ratio) or abs(final_to_current_ratio - 1.0) > 1.0e-5:
        raise RuntimeError("V14b final gradient norm failed current norm matching.")
    if projection_active and abs(cosine_post) > 1.0e-5:
        raise RuntimeError("V14b last1 projection failed orthogonality tolerance.")
    if not all(bool(torch.all(torch.isfinite(item)).item()) for item in combined):
        raise FloatingPointError("V14b combined gradient contains NaN/Inf.")
    return {
        "dot_raw": dot_raw,
        "cosine_raw": float(cosine_raw),
        "current_norm": current_norm,
        "last1_norm_raw": last1_norm_raw,
        "last1_norm_scaled": last1_scaled_norm,
        "last1_scale_factor": last_scale,
        "scaled_to_current_norm_ratio": scaled_to_current_ratio,
        "current_zero_fallback": current_zero_fallback,
        "projection_active": bool(projection_active),
        "last1_adjusted": last1_adjusted,
        "cosine_post_projection": float(cosine_post),
        "combined": combined,
        "combined_norm_before_clip": combined_norm,
        "final_to_current_norm_ratio": final_to_current_ratio,
        "mixed_norm": mixed_norm,
    }


def _explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float | None:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    variance = float(np.var(target))
    if variance <= 1.0e-12:
        return None
    return float(1.0 - np.var(target - prediction) / variance)


def _pearson_correlation(targets: np.ndarray, predictions: np.ndarray) -> float | None:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or len(target) < 2:
        return None
    if float(np.std(target)) <= 1.0e-12 or float(np.std(prediction)) <= 1.0e-12:
        return None
    return float(np.corrcoef(target, prediction)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        left_values.shape != right_values.shape
        or len(left_values) < 2
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        return None
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    if float(np.std(left_ranks)) <= 1.0e-12 or float(np.std(right_ranks)) <= 1.0e-12:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _full_information_trigger_surrogate(
    p_new: Any, p_old: Any, delta_scaled: Any, clip: float
) -> tuple[Any, Any, Any]:
    """Return per-state binary full-information PPO objective and action ratios."""

    epsilon = 1.0e-8
    old_update = p_old.clamp(epsilon, 1.0 - epsilon)
    new_update = p_new.clamp(epsilon, 1.0 - epsilon)
    old_keep = 1.0 - old_update
    new_keep = 1.0 - new_update
    advantage_update = old_keep * delta_scaled
    advantage_keep = -old_update * delta_scaled
    ratio_update = new_update / old_update
    ratio_keep = new_keep / old_keep
    objective_update = torch.minimum(
        ratio_update * advantage_update,
        ratio_update.clamp(1.0 - clip, 1.0 + clip) * advantage_update,
    )
    objective_keep = torch.minimum(
        ratio_keep * advantage_keep,
        ratio_keep.clamp(1.0 - clip, 1.0 + clip) * advantage_keep,
    )
    state_objective = old_update * objective_update + old_keep * objective_keep
    return state_objective, ratio_update, ratio_keep


def _binary_distribution_kl(p_old: Any, p_new: Any) -> Any:
    epsilon = 1.0e-8
    old_update = p_old.clamp(epsilon, 1.0 - epsilon)
    new_update = p_new.clamp(epsilon, 1.0 - epsilon)
    old_keep = (1.0 - p_old).clamp(epsilon, 1.0 - epsilon)
    new_keep = (1.0 - p_new).clamp(epsilon, 1.0 - epsilon)
    return (
        old_update * (torch.log(old_update) - torch.log(new_update))
        + old_keep * (torch.log(old_keep) - torch.log(new_keep))
    ).clamp_min(0.0)


def _capture_transaction_state(module: Any, optimizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    module_state = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return module_state, optimizer_state


def _restore_transaction_state(
    module: Any,
    optimizer: Any,
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    module_state, optimizer_state = snapshot
    module.load_state_dict(
        {key: value.detach().clone() for key, value in module_state.items()},
        strict=True,
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def _transactional_optimizer_update(
    module: Any,
    optimizer: Any,
    *,
    step_fn: Any,
    kl_fn: Any,
    hard_kl: float = TRIGGER_HARD_KL,
    step_scales: tuple[float, ...] = TRIGGER_BACKTRACKING_STEP_SCALES,
) -> dict[str, Any]:
    """Apply one optimizer update transactionally with exact-KL backtracking."""

    if not step_scales or any(not math.isfinite(scale) or scale <= 0.0 for scale in step_scales):
        raise ValueError("Transactional optimizer step scales must be finite and positive.")
    snapshot = _capture_transaction_state(module, optimizer)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    attempt_records: list[dict[str, Any]] = []
    rollback_count = 0
    max_rejected_kl_mean = 0.0
    max_rejected_kl_per_state = 0.0

    for attempt_index, scale in enumerate(step_scales, start=1):
        _restore_transaction_state(module, optimizer, snapshot)
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * float(scale)
        try:
            step_metadata = dict(step_fn())
            optimizer.step()
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr
            kl_mean, kl_max = kl_fn()
            kl_mean = float(kl_mean)
            kl_max = float(kl_max)
            if not math.isfinite(kl_mean) or not math.isfinite(kl_max):
                raise FloatingPointError("Transactional Trigger KL contains NaN/Inf.")
        except Exception:
            _restore_transaction_state(module, optimizer, snapshot)
            raise

        accepted = kl_mean <= float(hard_kl)
        attempt_records.append(
            {
                "attempt": int(attempt_index),
                "step_scale": float(scale),
                "kl_mean": kl_mean,
                "kl_max": kl_max,
                "accepted": bool(accepted),
                **step_metadata,
            }
        )
        if accepted:
            return {
                "accepted": True,
                "accepted_scale": float(scale),
                "accepted_kl_mean": kl_mean,
                "accepted_kl_max": kl_max,
                "max_rejected_kl_mean": max_rejected_kl_mean,
                "max_rejected_kl_per_state": max_rejected_kl_per_state,
                "rollback_count": rollback_count,
                "attempt_count": len(attempt_records),
                "attempt_records": attempt_records,
            }

        max_rejected_kl_mean = max(max_rejected_kl_mean, kl_mean)
        max_rejected_kl_per_state = max(max_rejected_kl_per_state, kl_max)
        rollback_count += 1
        _restore_transaction_state(module, optimizer, snapshot)

    _restore_transaction_state(module, optimizer, snapshot)
    return {
        "accepted": False,
        "accepted_scale": None,
        "accepted_kl_mean": 0.0,
        "accepted_kl_max": 0.0,
        "max_rejected_kl_mean": max_rejected_kl_mean,
        "max_rejected_kl_per_state": max_rejected_kl_per_state,
        "rollback_count": rollback_count,
        "attempt_count": len(attempt_records),
        "attempt_records": attempt_records,
    }


def _macro_batch(buffer: MacroRolloutBuffer) -> dict[str, np.ndarray]:
    items = buffer.transitions
    if not items:
        raise ValueError("Macro rollout buffer 不能为空。")
    batch_size = len(items)
    max_candidates = max(len(item.cache_candidate_masks) for item in items)
    feature_dim = int(items[0].cache_candidate_features.shape[-1])
    features = np.zeros((batch_size, max_candidates, feature_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_candidates), dtype=bool)
    for index, item in enumerate(items):
        count = len(item.cache_candidate_masks)
        if item.cache_candidate_features.shape != (count, feature_dim):
            raise ValueError("Macro candidate descriptor shape changed within rollout。")
        features[index, :count] = item.cache_candidate_features
        masks[index, :count] = item.cache_candidate_masks
    return {
        "observations": np.stack([item.observation for item in items]).astype(np.float32),
        "next_observations": np.stack([item.next_observation for item in items]).astype(np.float32),
        "candidate_features": features,
        "candidate_masks": masks,
        "duration_masks": np.stack([item.duration_masks for item in items]).astype(bool),
        "cache_actions": np.asarray([item.cache_action for item in items], dtype=np.int64),
        "duration_actions": np.asarray([item.duration_action for item in items], dtype=np.int64),
        "requested_durations": np.asarray(
            [item.requested_duration for item in items], dtype=np.int64
        ),
        "old_log_probs": np.asarray([item.old_macro_log_prob for item in items], dtype=np.float32),
        "old_trigger_log_probs": np.asarray(
            [item.old_cache_trigger_log_prob for item in items], dtype=np.float32
        ),
        "old_trigger_p_keep": np.asarray(
            [item.old_cache_trigger_p_keep for item in items], dtype=np.float64
        ),
        "old_trigger_p_update": np.asarray(
            [item.old_cache_trigger_p_update for item in items], dtype=np.float64
        ),
        "old_trigger_log_p_keep": np.asarray(
            [item.old_cache_trigger_log_p_keep for item in items], dtype=np.float64
        ),
        "old_trigger_log_p_update": np.asarray(
            [item.old_cache_trigger_log_p_update for item in items], dtype=np.float64
        ),
        "old_update_conditional_log_probs": np.asarray(
            [item.old_update_conditional_log_prob for item in items], dtype=np.float32
        ),
        "update_conditional_log_prob_valid": np.asarray(
            [item.update_conditional_log_prob_valid for item in items], dtype=bool
        ),
        "old_duration_log_probs": np.asarray(
            [item.old_duration_log_prob for item in items], dtype=np.float32
        ),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
        "old_next_values": np.asarray(
            [item.old_next_value for item in items], dtype=np.float32
        ),
        "rewards": np.asarray([item.frame_reward for item in items], dtype=np.float32),
        "durations": np.asarray([item.actual_duration for item in items], dtype=np.int64),
        "dones": np.asarray([item.done for item in items], dtype=bool),
        "trigger_teacher_valid": np.asarray(
            [item.trigger_teacher_valid for item in items], dtype=bool
        ),
        "trigger_learning_valid": np.asarray(
            [item.trigger_learning_valid for item in items], dtype=bool
        ),
        "trigger_credit_active": np.asarray(
            [item.trigger_credit_active for item in items], dtype=bool
        ),
        "trigger_teacher_q_keep": np.asarray(
            [0.0 if item.teacher_q_keep is None else item.teacher_q_keep for item in items],
            dtype=np.float64,
        ),
        "trigger_teacher_q_update": np.asarray(
            [0.0 if item.teacher_q_update is None else item.teacher_q_update for item in items],
            dtype=np.float64,
        ),
        "trigger_teacher_delta": np.asarray(
            [
                0.0
                if item.teacher_delta_update_keep is None
                else item.teacher_delta_update_keep
                for item in items
            ],
            dtype=np.float64,
        ),
        "trigger_update_standard_error": np.asarray(
            [item.trigger_update_standard_error for item in items], dtype=np.float64
        ),
        "trigger_confidence_ratio": np.asarray(
            [item.trigger_confidence_ratio for item in items], dtype=np.float64
        ),
        "trigger_actions": np.asarray(
            [int(item.cache_action > 0) for item in items], dtype=np.int64
        ),
        "macro_trigger_advantage": np.asarray(
            [item.trigger_advantage for item in items], dtype=np.float32
        ),
        "macro_trigger_advantage_raw": np.asarray(
            [item.trigger_advantage_raw for item in items], dtype=np.float32
        ),
        "teacher_delta_update_keep": np.asarray(
            [
                0.0
                if item.teacher_delta_update_keep is None
                else item.teacher_delta_update_keep
                for item in items
            ],
            dtype=np.float32,
        ),
        "macro_indices": np.arange(batch_size, dtype=np.int64),
    }


def _equal_realization_numpy_mean(values: Any, realization_ids: Any) -> float:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    group_array = np.asarray(realization_ids, dtype=np.int64).reshape(-1)
    if value_array.shape != group_array.shape or len(value_array) == 0:
        raise ValueError("Realization-weighted diagnostics require matching non-empty arrays.")
    group_means = [
        float(value_array[group_array == group].mean())
        for group in np.unique(group_array)
        if np.any(group_array == group)
    ]
    if not group_means or not np.isfinite(group_means).all():
        raise FloatingPointError("Realization-weighted diagnostic contains NaN/Inf.")
    return float(np.mean(group_means))


def _equal_realization_tensor_mean(values: Any, realization_ids: Any) -> Any:
    if torch is None:
        raise ImportError("Realization-weighted Trigger loss requires PyTorch.")
    group_tensor = torch.as_tensor(
        realization_ids, dtype=torch.long, device=values.device
    )
    if values.ndim != 1 or group_tensor.ndim != 1 or values.shape != group_tensor.shape:
        raise ValueError("Realization-weighted Trigger loss arrays must be matching vectors.")
    means = [values[group_tensor == group].mean() for group in torch.unique(group_tensor)]
    if not means:
        raise ValueError("Realization-weighted Trigger loss has no realization groups.")
    return torch.stack(means).mean()


def _micro_batch(buffer: MicroRolloutBuffer) -> dict[str, np.ndarray]:
    items = buffer.transitions
    if not items:
        raise ValueError("Micro rollout buffer 不能为空。")
    return {
        "observations": np.stack([item.observation for item in items]).astype(np.float32),
        "candidate_features": np.stack([item.uav_candidate_features for item in items]).astype(np.float32),
        "candidate_masks": np.stack([item.uav_candidate_masks for item in items]).astype(bool),
        "actions": np.stack([item.actions for item in items]).astype(np.int64),
        "conditions": np.stack([item.uav0_position_condition for item in items]).astype(np.float32),
        "old_log_probs": np.asarray([item.old_joint_log_prob for item in items], dtype=np.float32),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
        "rewards": np.asarray([item.reward for item in items], dtype=np.float32),
        "dones": np.asarray([item.done for item in items], dtype=bool),
    }


class PPOTrainer:
    """持有分离 Macro/Micro Actor-Critic、优化器与 episode update。"""

    checkpoint_version = CHECKPOINT_VERSION
    algorithm_id = ALGORITHM_ID

    def __init__(
        self,
        context: Any,
        *,
        device: str = "cpu",
        actor_learning_rate_override: float | None = None,
        macro_actor_learning_rate_override: float | None = None,
        critic_learning_rate_override: float | None = None,
        capture_macro_transition_trace: bool = False,
        stage1_trigger_only: bool = False,
        trigger_observation_normalization: dict[str, Any] | None = None,
        unified_cache_action: bool = False,
        cardinality_corrected_cache_action: bool = False,
        freeze_micro_updates: bool = False,
    ) -> None:
        _require_torch()
        self.context = context
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable。")
        self.config = context.cfg["algorithm"]["ppo"]
        self.unified_cache_action = bool(unified_cache_action)
        self.cardinality_corrected_cache_action = bool(
            cardinality_corrected_cache_action
        )
        self.freeze_micro_updates = bool(freeze_micro_updates)
        if self.cardinality_corrected_cache_action and not self.unified_cache_action:
            raise ValueError(
                "Cache cardinality correction requires the unified Cache action."
            )
        if self.unified_cache_action and stage1_trigger_only:
            raise ValueError("Unified Cache Action cannot use Trigger-only Stage-1 mode.")
        if self.unified_cache_action and trigger_observation_normalization is not None:
            raise ValueError("Unified Cache Action does not use Trigger observation normalization.")
        self.checkpoint_version = (
            UNIFIED_CACHE_CARDINALITY_CHECKPOINT_VERSION
            if self.cardinality_corrected_cache_action
            else UNIFIED_CACHE_CHECKPOINT_VERSION
            if self.unified_cache_action
            else CHECKPOINT_VERSION
        )
        self.algorithm_id = (
            UNIFIED_CACHE_CARDINALITY_ALGORITHM_ID
            if self.cardinality_corrected_cache_action
            else UNIFIED_CACHE_ALGORITHM_ID
            if self.unified_cache_action
            else ALGORITHM_ID
        )
        self.checkpoint_method = (
            UNIFIED_CACHE_CARDINALITY_CHECKPOINT_METHOD
            if self.cardinality_corrected_cache_action
            else UNIFIED_CACHE_CHECKPOINT_METHOD
            if self.unified_cache_action
            else CHECKPOINT_METHOD
        )
        self.capture_macro_transition_trace = bool(capture_macro_transition_trace)
        self.last_macro_transition_trace: dict[str, np.ndarray] | None = None
        self.stage1_trigger_only = bool(stage1_trigger_only)
        self.trigger_max_grad_norm = float(self.config["max_grad_norm"])
        if self.stage1_trigger_only and not math.isclose(
            self.trigger_max_grad_norm, TRIGGER_MAX_GRAD_NORM, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("V15-C2 Trigger gradient clip must be max_grad_norm=0.5.")
        self.initialization_metadata: dict[str, Any] = {}
        self.macro_critic_target_mode = str(
            self.config.get("macro_critic_target_mode", MACRO_CRITIC_TARGET_MODE)
        )
        if self.macro_critic_target_mode != MACRO_CRITIC_TARGET_MODE:
            raise ValueError(
                "V14b requires macro_critic_target_mode = two_step_smdp_td."
            )
        raw_critic_lambda = self.config.get("macro_critic_lambda_return_lambda")
        self.macro_critic_lambda_return_lambda = (
            None if raw_critic_lambda is None else float(raw_critic_lambda)
        )
        if self.macro_critic_lambda_return_lambda is not None and (
            not math.isfinite(self.macro_critic_lambda_return_lambda)
            or not 0.0 <= self.macro_critic_lambda_return_lambda <= 1.0
        ):
            raise ValueError(
                "macro_critic_lambda_return_lambda must be finite and in [0, 1]."
            )
        self.long_value_enabled = bool(self.config.get("long_value_enabled", False))
        self.long_value_target = str(
            self.config.get("long_value_target", "realized_return_to_go")
        )
        if not self.long_value_enabled:
            raise ValueError("architecture v15 requires long_value_enabled = true.")
        if self.long_value_target != "realized_return_to_go":
            raise ValueError(
                "Long-Value Critic target must be realized_return_to_go."
            )
        self.long_value_steps_per_update = int(
            self.config.get("long_value_steps_per_update", 50)
        )
        if self.long_value_steps_per_update != 50:
            raise ValueError("Long-Value Critic update budget is fixed at 50 steps.")
        self.long_value_max_grad_norm = float(
            self.config.get("long_value_max_grad_norm", 0.5)
        )
        if (
            not math.isfinite(self.long_value_max_grad_norm)
            or self.long_value_max_grad_norm <= 0.0
        ):
            raise ValueError("long_value_max_grad_norm must be finite and positive.")
        self.duration_candidates = macro_duration_candidates_from_project(context.cfg)
        tensor_spec = PrimitiveTensorSpec.create(context)
        macro_feature_dim = 2 * int(np.asarray(context.initial_cache).size) + 1
        micro_feature_dim = int(tensor_spec.uav_feature_dim) + 2
        self.hidden_sizes = tuple(int(value) for value in self.config["hidden_sizes"])
        self.policy = DualTimescaleActorCritic(
            int(context.observation_spec.observation_dim),
            macro_feature_dim,
            micro_feature_dim,
            self.duration_candidates,
            int(context.observation_spec.prediction_horizon),
            self.hidden_sizes,
            unified_cache_action=self.unified_cache_action,
            cardinality_corrected_cache_action=self.cardinality_corrected_cache_action,
        ).to(self.device)
        if trigger_observation_normalization is not None:
            self.policy.set_trigger_observation_normalization(
                trigger_observation_normalization["mean"],
                trigger_observation_normalization["std"],
                metadata=trigger_observation_normalization.get("metadata", {}),
            )
        if self.stage1_trigger_only and not self.policy.trigger_observation_normalization_active:
            raise ValueError("V15-C Stage-1 requires active Trigger-only observation normalization.")
        actor_lr = float(
            actor_learning_rate_override
            if actor_learning_rate_override is not None
            else self.config.get("actor_learning_rate", self.config["learning_rate"])
        )
        macro_actor_lr = float(
            macro_actor_learning_rate_override
            if macro_actor_learning_rate_override is not None
            else actor_lr
        )
        if not math.isfinite(macro_actor_lr) or macro_actor_lr <= 0.0:
            raise ValueError("macro actor learning rate must be finite and positive.")
        critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("critic_learning_rate", self.config["learning_rate"])
        )
        macro_critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("macro_critic_learning_rate", critic_lr)
        )
        micro_critic_lr = float(
            critic_learning_rate_override
            if critic_learning_rate_override is not None
            else self.config.get("micro_critic_learning_rate", critic_lr)
        )
        long_value_lr = float(self.config.get("long_value_learning_rate", 1.0e-3))
        if not math.isfinite(long_value_lr) or long_value_lr <= 0.0:
            raise ValueError("long_value_learning_rate must be finite and positive.")
        self.macro_optimizer = torch.optim.Adam(
            self.policy.macro_actor_parameters(),
            lr=macro_actor_lr,
            eps=1.0e-5,
        )
        self.macro_cache_trigger_actor_optimizer = torch.optim.Adam(
            self.policy.macro_cache_trigger_actor_parameters(),
            lr=macro_actor_lr,
            eps=1.0e-5,
        )
        self.macro_critic_optimizer = torch.optim.Adam(
            self.policy.macro_critic_parameters(),
            lr=macro_critic_lr,
            eps=1.0e-5,
        )
        macro_actor_parameter_ids = {
            id(parameter) for parameter in self.policy.macro_actor_parameters()
        }
        macro_critic_parameter_ids = {
            id(parameter) for parameter in self.policy.macro_critic_parameters()
        }
        trigger_parameter_ids = {
            id(parameter)
            for parameter in self.policy.macro_cache_trigger_actor_parameters()
        }
        if macro_actor_parameter_ids.intersection(macro_critic_parameter_ids):
            raise RuntimeError("V14 Macro Actor and scalar Critic parameters must be disjoint.")
        if trigger_parameter_ids.intersection(macro_actor_parameter_ids):
            raise RuntimeError("Independent Cache Trigger and Macro Actor parameters overlap.")
        if trigger_parameter_ids.intersection(macro_critic_parameter_ids):
            raise RuntimeError("Independent Cache Trigger and Main Critic parameters overlap.")
        other_actor_parameter_ids = {
            id(parameter)
            for parameter in (
                *self.policy.long_value_critic_parameters(),
                *self.policy.micro_actor_parameters(),
                *self.policy.micro_critic_parameters(),
            )
        }
        if trigger_parameter_ids.intersection(other_actor_parameter_ids):
            raise RuntimeError("Independent Cache Trigger overlaps another policy network.")
        self.micro_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.micro_actor_parameters(), "lr": actor_lr},
                {"params": self.policy.micro_critic_parameters(), "lr": micro_critic_lr},
            ],
            eps=1.0e-5,
        )
        self.long_value_optimizer = torch.optim.Adam(
            self.policy.long_value_critic_parameters(),
            lr=long_value_lr,
            eps=1.0e-5,
        )
        self.learning_rates = {
            "actor": actor_lr,
            "macro_actor": macro_actor_lr,
            "macro_cache_trigger_actor": macro_actor_lr,
            "micro_actor": actor_lr,
            "critic": critic_lr,
            "macro_critic": macro_critic_lr,
            "micro_critic": micro_critic_lr,
            "long_value": long_value_lr,
        }
        self.training_counters = {
            "completed_episodes": 0,
            "micro_optimizer_steps": 0,
            "macro_optimizer_steps": 0,
            "micro_actor_optimizer_steps": 0,
            "micro_critic_optimizer_steps": 0,
            "macro_actor_optimizer_steps": 0,
            "macro_cache_trigger_optimizer_steps": 0,
            "macro_critic_optimizer_steps": 0,
            "long_value_optimizer_steps": 0,
            "long_value_update_count": 0,
            "micro_update_count": 0,
            "macro_update_count": 0,
            "trigger_update_count": 0,
            "trigger_epoch_early_stop_count": 0,
            "trigger_epoch_accepted_count": 0,
            "trigger_epoch_rejected_count": 0,
            "trigger_optimizer_step_attempt_count": 0,
            "trigger_rollback_count": 0,
            "formal_non_trigger_optimizer_steps": 0,
            "environment_executions": 0,
            "matching_calls": 0,
            "kkt_calls": 0,
        }
        self.last1_macro_transition_buffer: dict[str, Any] | None = None

    def load_v14b_downstream_checkpoint(
        self,
        path: str | Path,
        *,
        expected_sha256: str = V14B_LATEST_CHECKPOINT_SHA256,
    ) -> dict[str, Any]:
        """Warm-start compatible mature V14b modules while keeping Trigger fresh."""

        checkpoint_path = Path(path).expanduser().resolve(strict=True)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest.lower() != str(expected_sha256).lower():
            raise ValueError(
                f"V14b checkpoint SHA256 mismatch: {digest} != {expected_sha256}."
            )
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("V14b checkpoint payload must be a mapping.")
        if int(payload.get("checkpoint_version", -1)) != V14B_CHECKPOINT_VERSION:
            raise ValueError("V14b source checkpoint_version must be 37.")
        if int(payload.get("architecture_version", -1)) != V14B_ARCHITECTURE_VERSION:
            raise ValueError("V14b source architecture_version must be 15.")
        counters = payload.get("training_counters")
        if not isinstance(counters, dict) or int(counters.get("completed_episodes", -1)) != 300:
            raise ValueError("V14b source checkpoint must be completed episode 300.")
        if payload.get("macro_critic_target_mode") != self.macro_critic_target_mode:
            raise ValueError("V14b source Main Critic target mode is incompatible.")
        if payload.get("long_value_enabled") is not self.long_value_enabled:
            raise ValueError("V14b source Long Critic enablement is incompatible.")
        if payload.get("long_value_target") != self.long_value_target:
            raise ValueError("V14b source Long Critic target is incompatible.")

        macro_state = payload.get("macro_state_dict")
        micro_state = payload.get("micro_state_dict")
        long_state = payload.get("long_value_critic_state_dict")
        if not all(isinstance(value, dict) for value in (macro_state, micro_state, long_state)):
            raise ValueError("V14b checkpoint is missing compatible downstream module states.")
        explicit_critic_state = payload.get("macro_critic_state_dict")
        if not isinstance(explicit_critic_state, dict):
            raise ValueError("V14b checkpoint is missing the explicit Main Critic state.")
        embedded_critic_state = {
            key.removeprefix("macro_critic."): value
            for key, value in macro_state.items()
            if key.startswith("macro_critic.")
        }
        if set(embedded_critic_state) != set(explicit_critic_state) or any(
            not torch.equal(embedded_critic_state[key], explicit_critic_state[key])
            for key in embedded_critic_state
        ):
            raise ValueError("V14b embedded and explicit Main Critic states disagree.")

        fresh_trigger_state = {
            key: value.detach().cpu().clone()
            for key, value in self.policy.macro_cache_trigger_actor.state_dict().items()
        }
        if not torch.equal(
            self.policy.macro_cache_trigger_actor[-1].weight,
            torch.zeros_like(self.policy.macro_cache_trigger_actor[-1].weight),
        ) or not torch.equal(
            self.policy.macro_cache_trigger_actor[-1].bias,
            torch.zeros_like(self.policy.macro_cache_trigger_actor[-1].bias),
        ):
            raise ValueError("Fresh V15-C Trigger output layer must initialize to equal logits.")
        self.policy.load_separated_state_dicts(
            macro_state,
            micro_state,
            long_state,
            fresh_trigger_state,
        )
        loaded_trigger_state = self.policy.macro_cache_trigger_actor.state_dict()
        if any(
            not torch.equal(loaded_trigger_state[key].detach().cpu(), value)
            for key, value in fresh_trigger_state.items()
        ):
            raise RuntimeError("V14b transplant unexpectedly changed the fresh Trigger.")
        metadata = {
            "downstream_warmstart_from_v14b": True,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": digest,
            "source_checkpoint_episode": int(counters["completed_episodes"]),
            "source_architecture_version": int(payload["architecture_version"]),
            "source_checkpoint_version": int(payload["checkpoint_version"]),
            "compatible_modules_loaded": [
                "Cache Candidate",
                "Duration",
                "Main Critic",
                "Long Critic",
                "Micro UAV Actor/Critic",
            ],
            "trigger_reinitialized": True,
            "trigger_initial_p_update": 0.5,
            "optimizer_states_transplanted": False,
            "training_counters_transplanted": False,
            "checkpoint_rng_restored": False,
        }
        self.initialization_metadata = metadata
        return metadata

    def load_v14b_unified_checkpoint(
        self,
        path: str | Path,
        *,
        expected_sha256: str = V14B_LATEST_CHECKPOINT_SHA256,
    ) -> dict[str, Any]:
        """Warm-start compatible modules while freshly initializing the unified action head."""

        if not self.unified_cache_action:
            raise RuntimeError("Unified V14b warm-start requires unified_cache_action=True.")
        checkpoint_path = Path(path).expanduser().resolve(strict=True)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest.lower() != str(expected_sha256).lower():
            raise ValueError(f"V14b checkpoint SHA256 mismatch: {digest} != {expected_sha256}.")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("V14b checkpoint payload must be a mapping.")
        if int(payload.get("checkpoint_version", -1)) != V14B_CHECKPOINT_VERSION:
            raise ValueError("V14b source checkpoint_version must be 37.")
        if int(payload.get("architecture_version", -1)) != V14B_ARCHITECTURE_VERSION:
            raise ValueError("V14b source architecture_version must be 15.")
        counters = payload.get("training_counters")
        if not isinstance(counters, dict) or int(counters.get("completed_episodes", -1)) != 300:
            raise ValueError("V14b source checkpoint must be completed episode 300.")
        if payload.get("macro_critic_target_mode") != self.macro_critic_target_mode:
            raise ValueError("V14b source Main Critic target mode is incompatible.")
        if payload.get("long_value_enabled") is not self.long_value_enabled:
            raise ValueError("V14b source Long Critic enablement is incompatible.")
        if payload.get("long_value_target") != self.long_value_target:
            raise ValueError("V14b source Long Critic target is incompatible.")

        macro_state = payload.get("macro_state_dict")
        micro_state = payload.get("micro_state_dict")
        long_state = payload.get("long_value_critic_state_dict")
        if not all(isinstance(value, dict) for value in (macro_state, micro_state, long_state)):
            raise ValueError("V14b checkpoint is missing compatible module states.")
        current_state = self.policy.state_dict()
        allowed_macro_prefixes = (
            "macro_actor_encoder.",
            "macro_cache_candidate_encoder.",
            "macro_duration_scorer.",
            "macro_critic.",
        )
        loaded_modules: list[str] = []
        for key, value in macro_state.items():
            if not key.startswith(allowed_macro_prefixes):
                continue
            if key not in current_state or tuple(current_state[key].shape) != tuple(value.shape):
                raise ValueError(f"V14b compatible Macro tensor shape mismatch at {key}.")
            current_state[key] = value.detach().to(device=current_state[key].device).clone()
        for key in micro_state:
            if key not in current_state or tuple(current_state[key].shape) != tuple(micro_state[key].shape):
                raise ValueError(f"V14b compatible Micro tensor shape mismatch at {key}.")
            current_state[key] = micro_state[key].detach().to(device=current_state[key].device).clone()
        for key, value in long_state.items():
            full_key = f"long_value_critic.{key}"
            if full_key not in current_state or tuple(current_state[full_key].shape) != tuple(value.shape):
                raise ValueError(f"V14b Long Critic tensor shape mismatch at {full_key}.")
            current_state[full_key] = value.detach().to(device=current_state[full_key].device).clone()
        self.policy.load_state_dict(current_state, strict=True)
        loaded_modules.extend(
            (
                "Macro shared observation encoder (reused by Duration)",
                "Macro candidate descriptor encoder (reused by Duration)",
                "Duration scorer",
                "Main Critic",
                "Long Critic",
                "Micro UAV Actor/Critic",
            )
        )
        metadata = {
            "downstream_warmstart_from_v14b": True,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": digest,
            "source_checkpoint_episode": int(counters["completed_episodes"]),
            "source_architecture_version": int(payload["architecture_version"]),
            "source_checkpoint_version": int(payload["checkpoint_version"]),
            "compatible_modules_loaded": loaded_modules,
            "candidate_selection_head_warmstart_used": False,
            "candidate_selection_head_fresh_init_reason": "KEEP candidate 0 becomes directly selectable; it was not in the V14b conditional UPDATE action distribution.",
            "candidate_shared_representation_loaded_for_duration": True,
            "trigger_reinitialized_and_bypassed": True,
            "optimizer_states_transplanted": False,
            "training_counters_transplanted": False,
            "checkpoint_rng_restored": False,
        }
        self.initialization_metadata = metadata
        return metadata

    @property
    def config_contract(self) -> dict[str, Any]:
        contract = {
            "observation_dim": int(self.policy.observation_dim),
            "macro_cache_feature_dim": int(self.policy.macro_cache_feature_dim),
            "micro_uav_feature_dim": int(self.policy.micro_uav_feature_dim),
            "duration_candidates": list(self.policy.duration_candidates),
            "prediction_horizon": int(self.policy.prediction_horizon),
            "hidden_sizes": list(self.hidden_sizes),
            "policy_schema": str(self.policy.architecture_id),
            "architecture_version": int(self.policy.architecture_version),
            "algorithm_id": self.algorithm_id,
            "macro_actor_advantage_mode": MACRO_ACTOR_ADVANTAGE_MODE,
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
            "cache_trigger_actor_architecture": "400-128-128-2 (SiLU), Trigger-only normalized Macro observation, independent",
            "cache_trigger_actor_initial_output": "final Linear weight and bias zero; equal KEEP/UPDATE logits",
            "cache_trigger_update_objective": "two-action full-information clipped PPO from exact binary old-policy KEEP/UPDATE advantages",
            "cache_trigger_transactional_trust_region": {
                "target_kl": TRIGGER_TARGET_KL,
                "hard_kl": TRIGGER_HARD_KL,
                "retry_step_scales": list(TRIGGER_BACKTRACKING_STEP_SCALES),
                "rollback_model_and_optimizer_state": True,
            },
            "cache_trigger_max_grad_norm": self.trigger_max_grad_norm,
            "trigger_observation_normalization_metadata": dict(
                self.policy.trigger_observation_normalization_metadata
            ),
            "trigger_observation_input_clip": self.policy.trigger_observation_input_clip,
            "cache_trigger_credit_source": "policy-marginalized duration reward difference; K=4 old conditional Candidate samples; no Critic bootstrap",
            "cache_trigger_candidate_samples": TRIGGER_CANDIDATE_SAMPLES,
            "cache_trigger_duration_candidates": list(self.policy.duration_candidates),
            "cache_trigger_uncertainty_gate": "active iff abs(A_trigger) > sample_std(K=4 candidate option returns)/sqrt(4)",
            "cache_trigger_advantage_scale": "active-transition median absolute deviation (MAD), divide by max(MAD, 1e-8), clip [-5, 5]",
            "cache_trigger_learning_rate": self.learning_rates["macro_cache_trigger_actor"],
            "cache_trigger_ppo_ratio": "separate KEEP and UPDATE probability ratios; no sampled-action or Candidate/Duration joint ratio",
            "cache_trigger_teacher_bootstrap": "none",
            "cache_trigger_updates_during_episode_collection": 0,
            "trigger_ppo_active": True,
            "cache_trigger_vs_macro_optimizer_parameter_overlap": 0,
            "macro_critic_value_loss_coefficient": float(
                self.config["value_loss_coefficient"]
            ),
            "macro_critic_architecture": "400-128-128-1 (SiLU)",
            "cache_trigger_teacher": "critic-independent paired option reward difference",
            "cache_trigger_counterfactual_branch_count_per_macro": "1 KEEP + 4 sampled UPDATE candidates, each run once to longest duration prefix",
            "cache_trigger_update_candidate_policy": "old-policy conditional UPDATE distribution",
            "stage1_trigger_only": self.stage1_trigger_only,
            "non_trigger_optimizer_steps_target": 0 if self.stage1_trigger_only else None,
            "trigger_observation_normalization_active": bool(
                self.policy.trigger_observation_normalization_active
            ),
            "macro_critic_optimizer": "single Adam",
            "macro_critic_optimizer_epsilon": 1.0e-5,
            "macro_critic_group_objective": "0.01 times raw current and last1 MSE",
            "macro_critic_conflict_projection": "last1 norm-matched; current-priority projection; final norm matched to current",
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "long_value_learning_rate": self.learning_rates["long_value"],
            "long_value_steps_per_update": self.long_value_steps_per_update,
            "long_value_max_grad_norm": self.long_value_max_grad_norm,
        }
        contract["unified_cache_action"] = self.unified_cache_action
        if self.unified_cache_action:
            contract.update(
                {
                    "cache_action_semantics": "single masked categorical over KEEP candidate 0 and legal UPDATE candidates; Duration conditioned on selected candidate",
                    "trigger_active": False,
                    "trigger_action_selection": False,
                    "trigger_log_probability": False,
                    "trigger_ppo_loss": False,
                    "trigger_optimizer_steps_target": 0,
                    "trigger_entropy_in_macro_entropy": False,
                    "trigger_in_macro_kl": False,
                    "trigger_teacher_used": False,
                    "trigger_counterfactual_branches": 0,
                    "cache_trigger_actor_architecture": "retained only for state-dict compatibility; bypassed",
                    "cache_trigger_update_objective": "inactive",
                    "cache_trigger_transactional_trust_region": "inactive",
                    "cache_trigger_max_grad_norm": 0.0,
                    "trigger_observation_normalization_active": False,
                    "cache_trigger_credit_source": "inactive",
                    "cache_trigger_candidate_samples": 0,
                    "cache_trigger_uncertainty_gate": "inactive",
                    "cache_trigger_advantage_scale": "inactive",
                    "cache_trigger_learning_rate": None,
                    "cache_trigger_ppo_ratio": "inactive",
                    "cache_trigger_teacher_bootstrap": "inactive",
                    "cache_trigger_teacher": "not used",
                    "cache_trigger_counterfactual_branch_count_per_macro": 0,
                    "cache_trigger_update_candidate_policy": "unified Cache Candidate categorical",
                    "macro_actor_advantage_mode": "formal Macro SMDP GAE; unified Candidate and Duration PPO",
                }
            )
        if self.cardinality_corrected_cache_action:
            contract["cardinality_corrected_cache_action"] = True
            contract["cache_candidate_probability_semantics"] = (
                "legal UPDATE logits subtract log of the legal nonzero candidate count"
            )
        return contract

    def update_from_episode(self, collection: EpisodeCollection) -> dict[str, Any]:
        """Update a completed rollout, optionally using V15-C Trigger-only Stage-1."""

        if not bool(collection.summary.get("counterfactual_state_restore_valid")):
            raise RuntimeError("COUNTERFACTUAL_STATE_RESTORE_VALID failed before PPO update.")
        if not bool(collection.summary.get("rollout_rng_unchanged")):
            raise RuntimeError("Formal rollout RNG changed during paired branch evaluation.")
        non_trigger_counter_keys = (
            "macro_actor_optimizer_steps",
            "macro_critic_optimizer_steps",
            "long_value_optimizer_steps",
            "micro_actor_optimizer_steps",
            "micro_critic_optimizer_steps",
        )
        non_trigger_before = sum(
            int(self.training_counters[key]) for key in non_trigger_counter_keys
        )
        if self.stage1_trigger_only:
            micro_metrics = {
                "micro_actor_optimizer_steps_this_update": 0,
                "micro_critic_optimizer_steps_this_update": 0,
                "micro_optimizer_frozen": True,
            }
            macro_metrics = self._update_trigger_only(collection.macro)
        elif self.freeze_micro_updates:
            micro_metrics = self._diagnose_micro_without_update(collection.micro)
            macro_metrics = self._update_macro(collection.macro)
        else:
            micro_metrics = self._update_micro(collection.micro)
            macro_metrics = self._update_macro(collection.macro)
        non_trigger_after = sum(
            int(self.training_counters[key]) for key in non_trigger_counter_keys
        )
        non_trigger_steps_this_update = non_trigger_after - non_trigger_before
        if self.stage1_trigger_only and non_trigger_steps_this_update != 0:
            raise RuntimeError(
                "V15-C Stage-1 changed a non-Trigger optimizer step counter."
            )
        self.training_counters["formal_non_trigger_optimizer_steps"] += int(
            non_trigger_steps_this_update
        )
        self.training_counters["completed_episodes"] += 1
        self.training_counters["environment_executions"] += int(
            collection.summary["environment_execution_count"]
        )
        self.training_counters["matching_calls"] += int(
            collection.summary["matching_call_count"]
        )
        self.training_counters["kkt_calls"] += int(collection.summary["kkt_call_count"])
        return {
            **micro_metrics,
            **macro_metrics,
            "formal_non_trigger_optimizer_steps_this_update": int(
                non_trigger_steps_this_update
            ),
        }

    def update_from_synchronized_episodes(
        self, collections: list[EpisodeCollection]
    ) -> dict[str, Any]:
        """Apply one Trigger-only update to a frozen-policy realization quartet."""

        if not self.stage1_trigger_only:
            raise RuntimeError("Synchronized Trigger updates require Stage-1 Trigger-only mode.")
        if len(collections) != 4:
            raise ValueError("V15-C4 synchronized updates require exactly four realizations.")
        for collection in collections:
            if not bool(collection.summary.get("counterfactual_state_restore_valid")):
                raise RuntimeError("COUNTERFACTUAL_STATE_RESTORE_VALID failed before PPO update.")
            if not bool(collection.summary.get("rollout_rng_unchanged")):
                raise RuntimeError("Formal rollout RNG changed during paired branch evaluation.")

        non_trigger_counter_keys = (
            "macro_actor_optimizer_steps",
            "macro_critic_optimizer_steps",
            "long_value_optimizer_steps",
            "micro_actor_optimizer_steps",
            "micro_critic_optimizer_steps",
        )
        non_trigger_before = sum(
            int(self.training_counters[key]) for key in non_trigger_counter_keys
        )
        merged_macro = MacroRolloutBuffer()
        realization_ids: list[int] = []
        for realization_index, collection in enumerate(collections):
            transitions = collection.macro.transitions
            if not transitions:
                raise RuntimeError("Synchronized realization contains no Macro transitions.")
            merged_macro.transitions.extend(transitions)
            realization_ids.extend([realization_index] * len(transitions))

        macro_metrics = self._update_trigger_only(
            merged_macro, realization_ids=np.asarray(realization_ids, dtype=np.int64)
        )
        non_trigger_after = sum(
            int(self.training_counters[key]) for key in non_trigger_counter_keys
        )
        non_trigger_steps = non_trigger_after - non_trigger_before
        if non_trigger_steps != 0:
            raise RuntimeError("V15-C4 changed a non-Trigger optimizer step counter.")
        self.training_counters["formal_non_trigger_optimizer_steps"] += int(non_trigger_steps)
        self.training_counters["completed_episodes"] += len(collections)
        self.training_counters["synchronized_trigger_updates"] = int(
            self.training_counters.get("synchronized_trigger_updates", 0) + 1
        )
        self.training_counters["synchronized_realizations"] = int(
            self.training_counters.get("synchronized_realizations", 0) + len(collections)
        )
        for collection in collections:
            self.training_counters["environment_executions"] += int(
                collection.summary["environment_execution_count"]
            )
            self.training_counters["matching_calls"] += int(
                collection.summary["matching_call_count"]
            )
            self.training_counters["kkt_calls"] += int(collection.summary["kkt_call_count"])
        return {
            **macro_metrics,
            "synchronized_realization_count": len(collections),
            "formal_non_trigger_optimizer_steps_this_update": int(non_trigger_steps),
        }

    def _update_trigger_only(
        self,
        buffer: MacroRolloutBuffer,
        *,
        realization_ids: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """V15-C full-information Trigger PPO with every downstream optimizer frozen."""

        if not self.stage1_trigger_only:
            raise RuntimeError("_update_trigger_only requires Stage-1 Trigger-only mode.")
        batch = _macro_batch(buffer)
        all_realization_ids = (
            np.zeros(len(batch["observations"]), dtype=np.int64)
            if realization_ids is None
            else np.asarray(realization_ids, dtype=np.int64).reshape(-1)
        )
        if all_realization_ids.shape != (len(batch["observations"]),):
            raise ValueError("Synchronized realization ids must match Macro transitions.")
        eligible_mask = batch["trigger_teacher_valid"] & batch["trigger_learning_valid"]
        eligible_indices = np.flatnonzero(eligible_mask)
        active_indices = np.flatnonzero(eligible_mask & batch["trigger_credit_active"])
        if len(eligible_indices) == 0:
            raise RuntimeError("V15-C rollout contained no eligible Trigger teacher states.")

        eligible_observations = torch.as_tensor(
            batch["observations"][eligible_indices], dtype=torch.float32, device=self.device
        )
        eligible_has_update = torch.as_tensor(
            batch["candidate_masks"][eligible_indices, 1:].any(axis=-1),
            dtype=torch.bool,
            device=self.device,
        )
        old_p_update_np = batch["old_trigger_p_update"][eligible_indices].astype(
            np.float32, copy=True
        )
        old_p_update = torch.as_tensor(
            old_p_update_np, dtype=torch.float32, device=self.device
        )
        active_positions = np.searchsorted(eligible_indices, active_indices)
        active_observations = torch.as_tensor(
            batch["observations"][active_indices], dtype=torch.float32, device=self.device
        )
        active_has_update = torch.as_tensor(
            batch["candidate_masks"][active_indices, 1:].any(axis=-1),
            dtype=torch.bool,
            device=self.device,
        )
        if len(active_indices) and not bool(torch.all(active_has_update).item()):
            raise RuntimeError("Active V15-C Trigger state has no legal UPDATE candidate.")

        old_p_active_np = old_p_update_np[active_positions]
        eligible_realization_ids = all_realization_ids[eligible_indices]
        active_realization_ids = all_realization_ids[active_indices]
        realization_groups = np.unique(eligible_realization_ids)
        if len(realization_groups) > 1:
            active_groups = np.unique(active_realization_ids)
            if not np.array_equal(active_groups, realization_groups):
                raise RuntimeError(
                    "Every synchronized realization must contribute active Trigger samples."
                )
        delta_active_np = batch["trigger_teacher_delta"][active_indices].astype(
            np.float64, copy=True
        )
        confidence_ratios = batch["trigger_confidence_ratio"][eligible_indices].astype(
            np.float64, copy=True
        )
        if not np.isfinite(old_p_update_np).all() or not np.isfinite(delta_active_np).all():
            raise FloatingPointError("V15-C old Trigger probabilities or DELTA contain NaN/Inf.")
        if len(active_indices):
            delta_median = float(np.median(delta_active_np))
            delta_mad = float(np.median(np.abs(delta_active_np - delta_median)))
        else:
            delta_median = 0.0
            delta_mad = 0.0
        delta_scale = max(delta_mad, TRIGGER_ADVANTAGE_EPSILON)
        delta_scaled_np = np.clip(
            delta_active_np / delta_scale,
            -TRIGGER_ADVANTAGE_CLIP,
            TRIGGER_ADVANTAGE_CLIP,
        ).astype(np.float32)
        old_p_active = torch.as_tensor(
            old_p_active_np, dtype=torch.float32, device=self.device
        )
        delta_scaled = torch.as_tensor(
            delta_scaled_np, dtype=torch.float32, device=self.device
        )

        all_observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        preclip_normalized_all_observations = (
            self.policy._preclip_normalized_trigger_observations(all_observations)
        )
        normalized_all_observations = self.policy._normalized_trigger_observations(
            all_observations
        )
        preclip_normalized_abs = (
            preclip_normalized_all_observations.detach()
            .abs()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        normalized_abs = (
            normalized_all_observations.detach().abs().cpu().numpy().astype(np.float64)
        )
        if not np.isfinite(preclip_normalized_abs).all() or not np.isfinite(normalized_abs).all():
            raise FloatingPointError("Normalized Trigger observations contain NaN/Inf.")
        feature_abs_max = normalized_abs.max(axis=0)
        preclip_feature_abs_max = preclip_normalized_abs.max(axis=0)
        low_variance_threshold = float(
            self.policy.trigger_observation_normalization_metadata.get(
                "low_variance_threshold", 1.0e-6
            )
        )
        low_variance_mask = (
            self.policy.trigger_observation_raw_std <= low_variance_threshold
        )
        low_variance_feature_indices = np.flatnonzero(low_variance_mask)
        expected_low_variance_count = self.policy.trigger_observation_normalization_metadata.get(
            "low_variance_feature_count",
            self.policy.trigger_observation_normalization_metadata.get(
                "standard_deviation_floor_feature_count"
            ),
        )
        if expected_low_variance_count is not None and int(
            expected_low_variance_count
        ) != len(low_variance_feature_indices):
            raise RuntimeError("Saved Trigger normalization low-variance feature count changed.")
        preclip_max = float(preclip_normalized_abs.max())
        postclip_max = float(normalized_abs.max())
        input_clip = self.policy.trigger_observation_input_clip
        clipped_value_fraction = (
            float(
                np.count_nonzero(np.abs(preclip_normalized_abs) > 10.0)
                / preclip_normalized_abs.size
            )
            if input_clip == (-10.0, 10.0)
            else 0.0
        )
        normalized_observation_diagnostics = {
            "trigger_normalized_obs_abs_max": postclip_max,
            "trigger_preclip_normalized_obs_abs_max": preclip_max,
            "trigger_postclip_normalized_obs_abs_max": postclip_max,
            "trigger_normalized_obs_clipped_value_fraction": clipped_value_fraction,
            "trigger_preclip_normalized_obs_p99_abs": float(
                np.percentile(preclip_normalized_abs, 99.0)
            ),
            "trigger_preclip_normalized_obs_mean_abs": float(
                preclip_normalized_abs.mean()
            ),
            "trigger_normalized_obs_p99_abs": float(np.percentile(normalized_abs, 99.0)),
            "trigger_normalized_obs_mean_abs": float(normalized_abs.mean()),
            "trigger_features_abs_gt_10_count": int(np.count_nonzero(feature_abs_max > 10.0)),
            "trigger_features_abs_gt_100_count": int(np.count_nonzero(feature_abs_max > 100.0)),
            "trigger_normalized_values_abs_gt_10_count": int(np.count_nonzero(normalized_abs > 10.0)),
            "trigger_normalized_values_abs_gt_100_count": int(np.count_nonzero(normalized_abs > 100.0)),
            "trigger_std_floor_feature_count": int(len(low_variance_feature_indices)),
            "trigger_low_variance_feature_count": int(len(low_variance_feature_indices)),
            "trigger_low_variance_feature_indices": low_variance_feature_indices.astype(int).tolist(),
            "trigger_std_floor_feature_indices": low_variance_feature_indices.astype(int).tolist(),
            "trigger_std_floor_features_abs_max": (
                float(preclip_feature_abs_max[low_variance_mask].max())
                if np.any(low_variance_mask)
                else 0.0
            ),
            "trigger_std_floor_features_abs_gt_100_count": int(
                np.count_nonzero(preclip_feature_abs_max[low_variance_mask] > 100.0)
            ),
            "trigger_normalization_numerical_risk": bool(preclip_max > 100.0),
            "trigger_low_variance_threshold": low_variance_threshold,
            "trigger_input_clip": (
                list(input_clip) if input_clip is not None else None
            ),
        }

        with torch.no_grad():
            before_distribution = self.policy.cache_trigger_distribution(
                eligible_observations, eligible_has_update
            )
            before_p_update = before_distribution.probs[:, 1]
            before_p_np = before_p_update.detach().cpu().numpy().astype(np.float64)
            before_entropy_np = (
                before_distribution.entropy().detach().cpu().numpy().astype(np.float64)
            )
            trigger_entropy_before = float(
                _equal_realization_numpy_mean(before_entropy_np, eligible_realization_ids)
            )
            trigger_p_update_before = _equal_realization_numpy_mean(
                before_p_np, eligible_realization_ids
            )
            trigger_p_update_std_before = _equal_realization_numpy_mean(
                [
                    float(before_p_np[eligible_realization_ids == group].std())
                    for group in realization_groups
                ],
                realization_groups,
            )

        clip = float(TRIGGER_FULL_INFORMATION_PPO_CLIP)
        entropy_coefficient = float(self.config["entropy_coefficient"])
        max_grad_norm = float(self.trigger_max_grad_norm)
        epoch_records: list[dict[str, Any]] = []
        epoch_early_stopped = False
        epoch_rejected_stop = False
        accepted_kl_means: list[float] = []
        accepted_kl_maxima: list[float] = []
        rejected_kl_means: list[float] = []
        rejected_kl_maxima: list[float] = []
        trigger_parameters = self.policy.macro_cache_trigger_actor_parameters()
        if len(active_indices):
            for epoch_index in range(int(self.config["update_epochs"])):
                step_metrics: dict[str, float] = {}

                def prepare_trigger_step() -> dict[str, float]:
                    active_distribution = self.policy.cache_trigger_distribution(
                        active_observations, active_has_update
                    )
                    p_new_active = active_distribution.probs[:, 1]
                    state_objective, ratio_update, ratio_keep = (
                        _full_information_trigger_surrogate(
                            p_new_active, old_p_active, delta_scaled, clip
                        )
                    )
                    entropy_per_state = active_distribution.entropy()
                    entropy = _equal_realization_tensor_mean(
                        entropy_per_state,
                        active_realization_ids,
                    )
                    state_objective_mean = _equal_realization_tensor_mean(
                        state_objective,
                        active_realization_ids,
                    )
                    loss = -state_objective_mean - entropy_coefficient * entropy
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            "Full-information Trigger loss contains NaN/Inf."
                        )
                    self.macro_cache_trigger_actor_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    pre_clip_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        trigger_parameters, max_grad_norm
                    )
                    pre_clip_norm = float(pre_clip_norm_tensor.detach().cpu())
                    post_clip_norm = _gradient_norm(trigger_parameters)
                    if not math.isfinite(pre_clip_norm) or not math.isfinite(post_clip_norm):
                        raise FloatingPointError("Trigger gradient norm contains NaN/Inf.")
                    clip_fraction_per_state = (
                        old_p_active * (torch.abs(ratio_update - 1.0) > clip).float()
                        + (1.0 - old_p_active)
                        * (torch.abs(ratio_keep - 1.0) > clip).float()
                    )
                    clip_fraction = float(
                        _equal_realization_tensor_mean(
                            clip_fraction_per_state, active_realization_ids
                        ).detach().cpu()
                    )
                    step_metrics.clear()
                    step_metrics.update(
                        {
                            "loss": float(loss.detach().cpu()),
                            "entropy": float(entropy.detach().cpu()),
                            "clip_fraction": clip_fraction,
                            "pre_clip_grad_norm": pre_clip_norm,
                            "post_clip_grad_norm": post_clip_norm,
                        }
                    )
                    return dict(step_metrics)

                def exact_old_batch_kl() -> tuple[float, float]:
                    with torch.no_grad():
                        current_distribution = self.policy.cache_trigger_distribution(
                            eligible_observations, eligible_has_update
                        )
                        current_p_update = current_distribution.probs[:, 1]
                        state_kl = _binary_distribution_kl(
                            old_p_update, current_p_update
                        )
                        return (
                            _equal_realization_numpy_mean(
                                state_kl.detach().cpu().numpy(), eligible_realization_ids
                            ),
                            float(state_kl.max().detach().cpu()),
                        )

                transaction = _transactional_optimizer_update(
                    self.policy.macro_cache_trigger_actor,
                    self.macro_cache_trigger_actor_optimizer,
                    step_fn=prepare_trigger_step,
                    kl_fn=exact_old_batch_kl,
                    hard_kl=TRIGGER_HARD_KL,
                    step_scales=TRIGGER_BACKTRACKING_STEP_SCALES,
                )
                self.training_counters["trigger_optimizer_step_attempt_count"] += int(
                    transaction["attempt_count"]
                )
                self.training_counters["trigger_rollback_count"] += int(
                    transaction["rollback_count"]
                )
                rejected_kl_means.extend(
                    float(item["kl_mean"])
                    for item in transaction["attempt_records"]
                    if not item["accepted"]
                )
                rejected_kl_maxima.extend(
                    float(item["kl_max"])
                    for item in transaction["attempt_records"]
                    if not item["accepted"]
                )
                if transaction["accepted"]:
                    self.training_counters["macro_cache_trigger_optimizer_steps"] += 1
                    self.training_counters["trigger_epoch_accepted_count"] += 1
                    accepted_kl_means.append(float(transaction["accepted_kl_mean"]))
                    accepted_kl_maxima.append(float(transaction["accepted_kl_max"]))
                    epoch_record = {
                        "epoch": int(epoch_index + 1),
                        "status": "ACCEPTED",
                        "accepted": True,
                        "accepted_scale": float(transaction["accepted_scale"]),
                        "attempt_count": int(transaction["attempt_count"]),
                        "rollback_count": int(transaction["rollback_count"]),
                        "accepted_kl_mean": float(transaction["accepted_kl_mean"]),
                        "accepted_kl_max": float(transaction["accepted_kl_max"]),
                        "max_rejected_kl_mean": float(transaction["max_rejected_kl_mean"]),
                        "max_rejected_kl_per_state": float(
                            transaction["max_rejected_kl_per_state"]
                        ),
                        "attempts": transaction["attempt_records"],
                        **step_metrics,
                    }
                    epoch_records.append(epoch_record)
                    if float(transaction["accepted_kl_mean"]) > TRIGGER_TARGET_KL:
                        epoch_early_stopped = True
                        self.training_counters["trigger_epoch_early_stop_count"] += 1
                        break
                else:
                    self.training_counters["trigger_epoch_rejected_count"] += 1
                    epoch_rejected_stop = True
                    epoch_records.append(
                        {
                            "epoch": int(epoch_index + 1),
                            "status": "REJECTED_ALL_RETRIES",
                            "accepted": False,
                            "accepted_scale": None,
                            "attempt_count": int(transaction["attempt_count"]),
                            "rollback_count": int(transaction["rollback_count"]),
                            "accepted_kl_mean": 0.0,
                            "accepted_kl_max": 0.0,
                            "max_rejected_kl_mean": float(
                                transaction["max_rejected_kl_mean"]
                            ),
                            "max_rejected_kl_per_state": float(
                                transaction["max_rejected_kl_per_state"]
                            ),
                            "attempts": transaction["attempt_records"],
                            **step_metrics,
                        }
                    )
                    break

        with torch.no_grad():
            final_distribution = self.policy.cache_trigger_distribution(
                eligible_observations, eligible_has_update
            )
            final_p_update = final_distribution.probs[:, 1]
            per_state_kl = _binary_distribution_kl(old_p_update, final_p_update)
            final_p_np = final_p_update.detach().cpu().numpy().astype(np.float64)
            final_entropy_np = (
                final_distribution.entropy().detach().cpu().numpy().astype(np.float64)
            )
            per_state_kl_np = per_state_kl.detach().cpu().numpy().astype(np.float64)
            trigger_kl_mean = _equal_realization_numpy_mean(
                per_state_kl_np, eligible_realization_ids
            )
            trigger_kl_max = float(per_state_kl.max().detach().cpu())
            trigger_entropy = _equal_realization_numpy_mean(
                final_entropy_np, eligible_realization_ids
            )
            trigger_p_update_after = _equal_realization_numpy_mean(
                final_p_np, eligible_realization_ids
            )
            trigger_p_update_state_std = _equal_realization_numpy_mean(
                [
                    float(final_p_np[eligible_realization_ids == group].std())
                    for group in realization_groups
                ],
                realization_groups,
            )
            if len(active_indices):
                active_final_p = final_p_update[
                    torch.as_tensor(active_positions, dtype=torch.long, device=self.device)
                ].detach().cpu().numpy()
                active_final_p = np.asarray(active_final_p, dtype=np.float64)
            else:
                active_final_p = np.empty(0, dtype=np.float64)

        active_count = int(len(active_indices))
        eligible_count = int(len(eligible_indices))
        positive_count = int(np.count_nonzero(delta_active_np > 0.0))
        negative_count = int(np.count_nonzero(delta_active_np < 0.0))
        active_fraction = _equal_realization_numpy_mean(
            [
                float(np.count_nonzero(eligible_realization_ids == group) and (
                    np.count_nonzero(active_realization_ids == group)
                    / np.count_nonzero(eligible_realization_ids == group)
                ))
                for group in realization_groups
            ],
            realization_groups,
        )
        delta_positive_fraction = _equal_realization_numpy_mean(
            [
                float(np.mean(delta_active_np[active_realization_ids == group] > 0.0))
                for group in np.unique(active_realization_ids)
            ],
            np.unique(active_realization_ids),
        ) if active_count else 0.0
        delta_negative_fraction = _equal_realization_numpy_mean(
            [
                float(np.mean(delta_active_np[active_realization_ids == group] < 0.0))
                for group in np.unique(active_realization_ids)
            ],
            np.unique(active_realization_ids),
        ) if active_count else 0.0
        confidence_median = float(np.median(confidence_ratios)) if len(confidence_ratios) else 0.0
        per_realization_spearman = {
            str(int(group)): _spearman_correlation(
                active_final_p[active_realization_ids == group],
                delta_active_np[active_realization_ids == group],
            )
            for group in np.unique(active_realization_ids)
        }
        finite_realization_spearman = [
            float(value)
            for value in per_realization_spearman.values()
            if value is not None and math.isfinite(float(value))
        ]
        per_realization_diagnostics = {
            str(int(group)): {
                "eligible_count": int(np.count_nonzero(eligible_realization_ids == group)),
                "active_count": int(np.count_nonzero(active_realization_ids == group)),
                "p_update_mean": float(final_p_np[eligible_realization_ids == group].mean()),
                "p_update_state_std": float(final_p_np[eligible_realization_ids == group].std()),
                "entropy": float(final_entropy_np[eligible_realization_ids == group].mean()),
                "active_fraction": float(
                    np.count_nonzero(active_realization_ids == group)
                    / np.count_nonzero(eligible_realization_ids == group)
                ),
                "delta_positive_fraction": float(
                    np.mean(delta_active_np[active_realization_ids == group] > 0.0)
                ),
                "delta_negative_fraction": float(
                    np.mean(delta_active_np[active_realization_ids == group] < 0.0)
                ),
                "policy_delta_spearman": per_realization_spearman[str(int(group))],
                "kl_mean": float(per_state_kl_np[eligible_realization_ids == group].mean()),
                "kl_max": float(per_state_kl_np[eligible_realization_ids == group].max()),
            }
            for group in realization_groups
        }
        result = {
            "macro_actor_optimizer_steps_this_update": 0,
            "macro_cache_trigger_optimizer_steps_this_update": int(
                sum(bool(record.get("accepted")) for record in epoch_records)
            ),
            "macro_critic_optimizer_steps_this_update": 0,
            "long_value_optimizer_steps_this_update": 0,
            "formal_non_trigger_optimizer_steps_this_update": 0,
            "macro_update_frozen": True,
            "trigger_ppo_active": bool(active_count),
            "trigger_ppo_epoch_count": len(epoch_records),
            "trigger_ppo_early_stop": bool(epoch_early_stopped or epoch_rejected_stop),
            "trigger_epoch_early_stop_count": int(epoch_early_stopped),
            "trigger_epoch_accepted": int(
                sum(bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_epoch_rejected": int(
                sum(not bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_epoch_accepted_count": int(
                sum(bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_epoch_rejected_count": int(
                sum(not bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_epoch_rollback_count": int(
                sum(int(record.get("rollback_count", 0)) for record in epoch_records)
            ),
            "trigger_epoch_records": epoch_records,
            "trigger_epoch_kl_means": [
                record["accepted_kl_mean"] for record in epoch_records
            ],
            "trigger_epoch_kl_maxima": [
                record["accepted_kl_max"] for record in epoch_records
            ],
            "trigger_kl_target": TRIGGER_TARGET_KL,
            "trigger_kl_hard_limit": TRIGGER_HARD_KL,
            "trigger_backtracking_step_scales": list(
                TRIGGER_BACKTRACKING_STEP_SCALES
            ),
            "trigger_kl_accepted_max": max(accepted_kl_means, default=0.0),
            "trigger_kl_accepted_per_state_max": max(accepted_kl_maxima, default=0.0),
            "trigger_kl_rejected_max": max(rejected_kl_means, default=0.0),
            "trigger_kl_rejected_per_state_max": max(rejected_kl_maxima, default=0.0),
            "trigger_epoch_accepted_count_this_update": int(
                sum(bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_epoch_rejected_count_this_update": int(
                sum(not bool(record.get("accepted")) for record in epoch_records)
            ),
            "trigger_rollback_count_this_update": int(
                sum(int(record.get("rollback_count", 0)) for record in epoch_records)
            ),
            "trigger_optimizer_step_attempts_this_update": int(
                sum(int(record.get("attempt_count", 0)) for record in epoch_records)
            ),
            "trigger_transactional_kl_rollback_active": True,
            "trigger_grad_clip_active": True,
            "trigger_max_grad_norm": max_grad_norm,
            "trigger_preclip_grad_norm": max(
                (float(record.get("pre_clip_grad_norm", 0.0)) for record in epoch_records),
                default=0.0,
            ),
            "trigger_postclip_grad_norm": max(
                (float(record.get("post_clip_grad_norm", 0.0)) for record in epoch_records),
                default=0.0,
            ),
            "trigger_entropy_before": trigger_entropy_before,
            "trigger_entropy": trigger_entropy,
            "trigger_p_update_before": trigger_p_update_before,
            "trigger_p_update_std_before": trigger_p_update_std_before,
            "trigger_p_update_after": trigger_p_update_after,
            "trigger_p_update_mean": trigger_p_update_after,
            "trigger_p_keep_after": 1.0 - trigger_p_update_after,
            "trigger_p_update_std_after": trigger_p_update_state_std,
            "trigger_p_update_state_std": trigger_p_update_state_std,
            "trigger_kl_mean": trigger_kl_mean,
            "trigger_kl_max": trigger_kl_max,
            "trigger_approx_kl": trigger_kl_mean,
            "trigger_clip_fraction": float(
                np.mean([record["clip_fraction"] for record in epoch_records])
                if epoch_records else 0.0
            ),
            "trigger_credit_eligible_count": eligible_count,
            "trigger_credit_active_count": active_count,
            "trigger_credit_active_fraction": active_fraction,
            "trigger_advantage_scale": float(delta_scale),
            "trigger_delta_raw_mean": (
                _equal_realization_numpy_mean(
                    [
                        float(delta_active_np[active_realization_ids == group].mean())
                        for group in np.unique(active_realization_ids)
                    ],
                    np.unique(active_realization_ids),
                )
                if active_count
                else 0.0
            ),
            "trigger_delta_raw_std": (
                _equal_realization_numpy_mean(
                    [
                        float(delta_active_np[active_realization_ids == group].std())
                        for group in np.unique(active_realization_ids)
                    ],
                    np.unique(active_realization_ids),
                )
                if active_count
                else 0.0
            ),
            "trigger_delta_raw_median": delta_median,
            "trigger_delta_raw_mad": delta_mad,
            "sync_batch_raw_delta_mad": delta_mad,
            "trigger_advantage_scaled_mean": float(np.mean(delta_scaled_np)) if active_count else 0.0,
            "trigger_advantage_scaled_std": float(np.std(delta_scaled_np)) if active_count else 0.0,
            "trigger_advantage_scaled_min": float(np.min(delta_scaled_np)) if active_count else 0.0,
            "trigger_advantage_scaled_max": float(np.max(delta_scaled_np)) if active_count else 0.0,
            "trigger_confidence_ratio_median": confidence_median,
            "trigger_confidence_ratio_p10": (
                float(np.percentile(confidence_ratios, 10.0)) if len(confidence_ratios) else 0.0
            ),
            "trigger_confidence_ratio_p90": (
                float(np.percentile(confidence_ratios, 90.0)) if len(confidence_ratios) else 0.0
            ),
            "trigger_delta_positive_count": positive_count,
            "trigger_delta_negative_count": negative_count,
            "trigger_delta_positive_fraction": delta_positive_fraction,
            "trigger_delta_negative_fraction": delta_negative_fraction,
            "trigger_positive_advantage_fraction": delta_positive_fraction,
            "trigger_negative_advantage_fraction": delta_negative_fraction,
            "trigger_policy_delta_spearman": (
                float(np.mean(finite_realization_spearman))
                if len(finite_realization_spearman) == len(per_realization_spearman)
                and finite_realization_spearman
                else None
            ),
            "pooled_trigger_policy_delta_spearman": _spearman_correlation(
                active_final_p, delta_active_np
            ),
            "realization_policy_delta_spearman": per_realization_spearman,
            "realization_trigger_diagnostics": per_realization_diagnostics,
            "realization_equal_weighting": bool(len(realization_groups) > 1),
            "trigger_policy_delta_spearman_active_count": active_count,
            "trigger_old_policy_probabilities_used": True,
            "trigger_full_information_binary_ppo": True,
            "trigger_entropy_regularization_coefficient": entropy_coefficient,
            "trigger_critic_bootstrap_used": False,
            **normalized_observation_diagnostics,
        }
        if self.training_counters["trigger_update_count"] >= 0:
            self.training_counters["trigger_update_count"] += 1
        return result

    def _diagnose_micro_without_update(
        self, buffer: MicroRolloutBuffer
    ) -> dict[str, Any]:
        """Compute frozen Micro rollout diagnostics without backward or optimizer calls."""

        batch = _micro_batch(buffer)
        count = len(batch["rewards"])
        values = batch["old_values"].astype(np.float64)
        advantages, returns = compute_gae(
            batch["rewards"],
            values,
            batch["dones"],
            0.0,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized, adv_mean, adv_std = _normalize_advantages(advantages)
        batch["advantages"] = normalized
        batch["returns"] = returns
        minibatches = min(int(self.config["num_minibatches"]), count)
        if minibatches <= 0:
            raise ValueError("Micro rollout 不能为空。")
        records: list[dict[str, float]] = []
        predictions = np.empty(count, dtype=np.float64)
        indices = np.arange(count, dtype=np.int64)
        for index_group in np.array_split(indices, minibatches):
            if len(index_group) == 0:
                continue
            tensors = self._tensor_batch(batch, index_group)
            with torch.no_grad():
                evaluated = self.policy.evaluate_micro_joint(
                    tensors["observations"],
                    tensors["candidate_features"],
                    tensors["candidate_masks"],
                    tensors["actions"],
                    tensors["conditions"],
                )
                log_ratio = evaluated.log_prob - tensors["old_log_probs"]
                ratio = torch.exp(log_ratio)
                advantage = tensors["advantages"]
                actor_loss = -torch.minimum(
                    ratio * advantage,
                    torch.clamp(
                        ratio,
                        1.0 - float(self.config["clip_epsilon"]),
                        1.0 + float(self.config["clip_epsilon"]),
                    )
                    * advantage,
                ).mean()
                value_loss = torch.mean(
                    (evaluated.value - tensors["returns"]) ** 2
                )
                entropy = evaluated.entropy.mean()
                approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                clip_fraction = torch.mean(
                    (
                        torch.abs(ratio - 1.0)
                        > float(self.config["clip_epsilon"])
                    ).float()
                )
                scalar_metrics = (
                    actor_loss,
                    value_loss,
                    entropy,
                    approx_kl,
                    clip_fraction,
                    evaluated.log_prob,
                    evaluated.value,
                )
                if not all(bool(torch.isfinite(value).all().item()) for value in scalar_metrics):
                    raise FloatingPointError(
                        "Frozen Micro diagnostics contain NaN/Inf."
                    )
                predictions[index_group] = (
                    evaluated.value.detach().cpu().numpy().astype(np.float64)
                )
                records.append(
                    {
                        "actor_loss": float(actor_loss.cpu()),
                        "value_loss": float(value_loss.cpu()),
                        "entropy": float(entropy.cpu()),
                        "approx_kl": float(approx_kl.cpu()),
                        "clip_fraction": float(clip_fraction.cpu()),
                    }
                )
        result = self._aggregate_update_metrics("micro", records)
        result.update(
            {
                "micro_actor_optimizer_steps_this_update": 0,
                "micro_critic_optimizer_steps_this_update": 0,
                "micro_optimizer_frozen": True,
                "micro_backward_calls": 0,
                "micro_transition_count": int(count),
                "micro_adv_mean": float(adv_mean),
                "micro_adv_std": float(adv_std),
                "micro_normalized_adv_mean": float(np.mean(normalized)),
                "micro_normalized_adv_std": float(np.std(normalized)),
                "micro_value_prediction_mean": float(np.mean(predictions)),
                "micro_value_prediction_std": float(np.std(predictions)),
                "micro_value_target_mean": float(np.mean(returns)),
                "micro_value_target_std": float(np.std(returns)),
                "micro_value_target_prediction_corr": _pearson_correlation(
                    returns, predictions
                ),
                "micro_value_rmse": float(
                    np.sqrt(np.mean((returns - predictions) ** 2))
                ),
                "micro_explained_variance": _explained_variance(
                    returns, predictions
                ),
                "micro_early_stop": False,
            }
        )
        return result

    def _update_micro(self, buffer: MicroRolloutBuffer) -> dict[str, Any]:
        batch = _micro_batch(buffer)
        count = len(batch["rewards"])
        values = batch["old_values"].astype(np.float64)
        advantages, returns = compute_gae(
            batch["rewards"],
            values,
            batch["dones"],
            0.0,
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized, adv_mean, adv_std = _normalize_advantages(advantages)
        batch["advantages"] = normalized
        batch["returns"] = returns
        minibatches = int(self.config["num_minibatches"])
        if count < minibatches:
            minibatches = count
        if minibatches <= 0:
            raise ValueError("Micro rollout 不能为空。")
        epochs = int(self.config["update_epochs"])
        clip = float(self.config["clip_epsilon"])
        value_coefficient = float(self.config["value_loss_coefficient"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        target_kl = float(self.config["target_kl"])
        max_grad_norm = float(self.config["max_grad_norm"])
        records: list[dict[str, float]] = []
        stopped = False
        for _epoch in range(epochs):
            indices = np.random.permutation(count)
            for index_group in np.array_split(indices, minibatches):
                if len(index_group) == 0:
                    continue
                tensors = self._tensor_batch(batch, index_group)
                evaluated = self.policy.evaluate_micro_joint(
                    tensors["observations"],
                    tensors["candidate_features"],
                    tensors["candidate_masks"],
                    tensors["actions"],
                    tensors["conditions"],
                )
                log_ratio = evaluated.log_prob - tensors["old_log_probs"]
                ratio = torch.exp(log_ratio)
                advantage = tensors["advantages"]
                actor_loss = -torch.minimum(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage,
                ).mean()
                value_loss = torch.mean((evaluated.value - tensors["returns"]) ** 2)
                entropy = evaluated.entropy.mean()
                loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
                self.micro_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                micro_actor_parameters = self.policy.micro_actor_parameters()
                micro_critic_parameters = self.policy.micro_critic_parameters()
                micro_actor_before_clip = float(
                    torch.nn.utils.clip_grad_norm_(micro_actor_parameters, max_grad_norm)
                    .detach()
                    .cpu()
                )
                micro_actor_after_clip = _gradient_norm(micro_actor_parameters)
                micro_critic_before_clip = float(
                    torch.nn.utils.clip_grad_norm_(micro_critic_parameters, max_grad_norm)
                    .detach()
                    .cpu()
                )
                micro_critic_after_clip = _gradient_norm(micro_critic_parameters)
                self.micro_optimizer.step()
                self.training_counters["micro_optimizer_steps"] += 1
                self.training_counters["micro_actor_optimizer_steps"] += 1
                self.training_counters["micro_critic_optimizer_steps"] += 1
                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip).float())
                records.append(
                    {
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "actor_grad_norm_before_clip": micro_actor_before_clip,
                        "actor_grad_norm_after_clip": micro_actor_after_clip,
                        "critic_grad_norm_before_clip": micro_critic_before_clip,
                        "critic_grad_norm_after_clip": micro_critic_after_clip,
                    }
                )
                if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                    stopped = True
                    break
            if stopped:
                break
        self.training_counters["micro_update_count"] += 1
        result = self._aggregate_update_metrics("micro", records)
        with torch.no_grad():
            predictions = (
                self.policy.micro_value(
                    torch.as_tensor(
                        batch["observations"], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
        result.update(
            {
                "micro_actor_optimizer_steps_this_update": len(records),
                "micro_critic_optimizer_steps_this_update": len(records),
                "micro_adv_mean": float(adv_mean),
                "micro_adv_std": float(adv_std),
                "micro_normalized_adv_mean": float(np.mean(normalized)),
                "micro_normalized_adv_std": float(np.std(normalized)),
                "micro_value_prediction_mean": float(np.mean(predictions)),
                "micro_value_prediction_std": float(np.std(predictions)),
                "micro_value_target_mean": float(np.mean(returns)),
                "micro_value_target_std": float(np.std(returns)),
                "micro_value_target_prediction_corr": _pearson_correlation(
                    returns, predictions
                ),
                "micro_value_rmse": float(
                    np.sqrt(np.mean((returns - predictions) ** 2))
                ),
                "micro_explained_variance": _explained_variance(
                    returns, predictions
                ),
                "micro_early_stop": bool(stopped),
            }
        )
        return result

    def _update_macro(self, buffer: MacroRolloutBuffer) -> dict[str, Any]:
        batch = _macro_batch(buffer)
        count = len(batch["rewards"])
        episode_index = int(self.training_counters["completed_episodes"]) + 1
        last1_buffer = self.last1_macro_transition_buffer
        if last1_buffer is not None and int(last1_buffer["episode_index"]) != episode_index - 1:
            raise RuntimeError("V14 last1 buffer must contain exactly the preceding episode.")
        next_values = batch["old_next_values"].astype(np.float32, copy=True)
        values = batch["old_values"].astype(np.float64)
        macro_actor_advantage, _gae_value_targets = compute_macro_smdp_gae(
            batch["rewards"],
            values,
            next_values,
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
            float(self.config["gae_lambda"]),
        )
        normalized_actor_advantage, adv_mean, adv_std = _normalize_advantages(
            macro_actor_advantage
        )
        current_observations = batch["observations"].astype(np.float32, copy=True)
        last1_observations = (
            None
            if last1_buffer is None
            else np.asarray(last1_buffer["observations"], dtype=np.float32).copy()
        )
        # Reuse exact rollout-frozen values to avoid batch-kernel round-off in
        # the two-step SMDP target.
        frozen_current_values = batch["old_values"].astype(np.float32, copy=True)
        with torch.no_grad():
            frozen_last1_values = (
                None
                if last1_observations is None
                else self.policy.macro_value(
                    torch.as_tensor(
                        last1_observations, dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
        macro_critic_target = build_macro_n2_target(
            batch["rewards"],
            frozen_current_values,
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
        ).astype(np.float32)
        last1_target = None
        if last1_buffer is not None:
            assert frozen_last1_values is not None
            last1_target = build_macro_n2_target(
                last1_buffer["macro_rewards"],
                frozen_last1_values,
                last1_buffer["actual_durations"],
                last1_buffer["dones"],
                float(self.config["gamma"]),
            ).astype(np.float32)
        realized_return_to_go = _realized_smdp_return_to_go(
            batch["rewards"],
            batch["durations"],
            batch["dones"],
            float(self.config["gamma"]),
        )
        self.last_macro_transition_trace = None
        if self.capture_macro_transition_trace:
            self.last_macro_transition_trace = {
                "observation": batch["observations"].astype(np.float32, copy=True),
                "next_observation": batch["next_observations"].astype(np.float32, copy=True),
                "macro_reward": batch["rewards"].astype(np.float32, copy=True),
                "requested_duration": batch["requested_durations"].astype(np.int64, copy=True),
                "actual_duration": batch["durations"].astype(np.int64, copy=True),
                "done": batch["dones"].astype(bool, copy=True),
                "old_value": frozen_current_values.copy(),
                "old_next_value": next_values.astype(np.float32, copy=True),
                "critic_training_target": macro_critic_target.astype(np.float32, copy=True),
                "realized_return_to_go": realized_return_to_go.astype(np.float32, copy=True),
            }
            n2_bootstrap_old_value = np.zeros(count, dtype=np.float32)
            n2_bootstrap_frame_index = np.full(count, -1, dtype=np.int64)
            for index in range(count):
                if batch["dones"][index] or batch["dones"][index + 1]:
                    continue
                n2_bootstrap_old_value[index] = frozen_current_values[index + 2]
                n2_bootstrap_frame_index[index] = index + 2
            self.last_macro_transition_trace.update(
                {
                    "n2_target": macro_critic_target.astype(np.float32, copy=True),
                    "n2_bootstrap_old_value": n2_bootstrap_old_value,
                    "n2_bootstrap_frame_index": n2_bootstrap_frame_index,
                }
            )
        batch["macro_critic_targets"] = macro_critic_target.copy()
        selected_groups = batch["cache_actions"] > 0
        old_p_keep = batch["old_trigger_p_keep"].astype(np.float64)
        old_p_update = batch["old_trigger_p_update"].astype(np.float64)
        eligible_trigger_mask = (
            batch["trigger_teacher_valid"] & batch["trigger_learning_valid"]
        )
        active_trigger_mask = eligible_trigger_mask & batch["trigger_credit_active"]
        eligible_trigger_indices = np.flatnonzero(eligible_trigger_mask)
        active_trigger_indices = np.flatnonzero(active_trigger_mask)
        trigger_raw_advantages = batch["trigger_teacher_delta"][eligible_trigger_indices]
        active_trigger_raw_advantages = batch["trigger_teacher_delta"][active_trigger_indices]
        if not np.isfinite(trigger_raw_advantages).all() or not np.isfinite(
            batch["trigger_confidence_ratio"][eligible_trigger_indices]
        ).all():
            raise FloatingPointError("Critic-independent Trigger targets contain NaN/Inf.")
        trigger_raw_median = (
            float(np.median(active_trigger_raw_advantages))
            if len(active_trigger_raw_advantages)
            else 0.0
        )
        trigger_raw_mad = (
            float(
                np.median(np.abs(active_trigger_raw_advantages - trigger_raw_median))
            )
            if len(active_trigger_raw_advantages)
            else 0.0
        )
        trigger_advantage_scale = max(trigger_raw_mad, TRIGGER_ADVANTAGE_EPSILON)
        trigger_scaled_advantages = np.clip(
            active_trigger_raw_advantages / trigger_advantage_scale,
            -TRIGGER_ADVANTAGE_CLIP,
            TRIGGER_ADVANTAGE_CLIP,
        ).astype(np.float32)
        batch["macro_trigger_advantage"] = np.zeros(count, dtype=np.float32)
        batch["macro_trigger_advantage"][active_trigger_indices] = trigger_scaled_advantages
        trigger_confidence_ratios = batch["trigger_confidence_ratio"][
            eligible_trigger_indices
        ]
        trigger_active_fraction = float(
            len(active_trigger_indices) / max(len(eligible_trigger_indices), 1)
        )
        clip = float(self.config["clip_epsilon"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        target_kl = float(self.config["target_kl"])
        max_grad_norm = float(self.config["max_grad_norm"])
        critic_steps = int(self.config.get("macro_critic_steps_per_update", 50))
        if critic_steps != 50:
            raise ValueError("V14 Macro Critic update budget is fixed at 50 iterations.")
        value_loss_coefficient = float(self.config["value_loss_coefficient"])
        if not math.isfinite(value_loss_coefficient) or value_loss_coefficient <= 0.0:
            raise ValueError("value_loss_coefficient must be finite and positive.")
        critic_fit = self._update_macro_critic_only(
            current_observations,
            macro_critic_target,
            last1_observations=last1_observations,
            last1_targets=last1_target,
            steps=critic_steps,
            value_loss_coefficient=value_loss_coefficient,
            max_grad_norm=max_grad_norm,
        )
        batch["macro_actor_advantage"] = normalized_actor_advantage
        records: list[dict[str, float]] = []
        stopped = False
        all_indices = np.arange(count, dtype=np.int64)
        max_decomposition_error = 0.0
        rollout_logprob_max_error = 0.0
        for _epoch in range(int(self.config["update_epochs"])):
            tensors = self._tensor_batch(batch, all_indices)
            components = self.policy.evaluate_macro_components(
                tensors["observations"],
                tensors["candidate_features"],
                tensors["candidate_masks"],
                tensors["cache_actions"],
                tensors["duration_masks"],
                tensors["duration_actions"],
            )
            decomposition_error = float(
                components.joint_logprob_decomposition_error.max().detach().cpu()
            )
            max_decomposition_error = max(max_decomposition_error, decomposition_error)
            if self.unified_cache_action and _epoch == 0:
                rollout_logprob_max_error = float(
                    torch.max(
                        torch.abs(components.joint_log_prob - tensors["old_log_probs"])
                    ).detach().cpu()
                )
                if not math.isfinite(rollout_logprob_max_error) or rollout_logprob_max_error > 5.0e-6:
                    raise RuntimeError(
                        "Unified sampled Macro log-probability does not match the stored rollout value: "
                        f"{rollout_logprob_max_error:.9g}."
                    )
            if self.unified_cache_action:
                old_candidate_log_probs = (
                    tensors["old_log_probs"] - tensors["old_duration_log_probs"]
                )
                candidate_log_ratio = components.cache_log_prob - old_candidate_log_probs
                candidate_ratio = torch.exp(candidate_log_ratio)
                candidate_adv = tensors["macro_actor_advantage"]
                candidate_loss = _clipped_ppo_surrogate_loss(
                    candidate_ratio, candidate_adv, clip
                )
                candidate_kl = torch.mean(
                    (candidate_ratio - 1.0) - candidate_log_ratio
                )
                candidate_clip_fraction = torch.mean(
                    (torch.abs(candidate_ratio - 1.0) > clip).float()
                )
            else:
                update_mask = tensors["update_conditional_log_prob_valid"].bool()
                if bool(torch.any(update_mask).item()):
                    candidate_log_ratio = (
                        components.conditional_update_log_prob[update_mask]
                        - tensors["old_update_conditional_log_probs"][update_mask]
                    )
                    candidate_ratio = torch.exp(candidate_log_ratio)
                    candidate_adv = tensors["macro_actor_advantage"][update_mask]
                    candidate_loss = _clipped_ppo_surrogate_loss(
                        candidate_ratio, candidate_adv, clip
                    )
                    candidate_kl = torch.mean(
                        (candidate_ratio - 1.0) - candidate_log_ratio
                    )
                    candidate_clip_fraction = torch.mean(
                        (torch.abs(candidate_ratio - 1.0) > clip).float()
                    )
                else:
                    candidate_log_ratio = components.conditional_update_log_prob[:0]
                    candidate_loss = components.conditional_update_log_prob.sum() * 0.0
                    candidate_kl = candidate_loss.detach()
                    candidate_clip_fraction = candidate_loss.detach()

            duration_log_ratio = (
                components.duration_log_prob - tensors["old_duration_log_probs"]
            )
            duration_ratio = torch.exp(duration_log_ratio)
            duration_adv = tensors["macro_actor_advantage"]
            duration_loss = _clipped_ppo_surrogate_loss(
                duration_ratio, duration_adv, clip
            )

            if self.unified_cache_action:
                macro_policy_log_prob = components.cache_log_prob + components.duration_log_prob
                old_macro_policy_log_prob = tensors["old_log_probs"]
            else:
                macro_policy_log_prob = (
                    components.conditional_update_log_prob + components.duration_log_prob
                )
                old_macro_policy_log_prob = (
                    tensors["old_update_conditional_log_probs"]
                    + tensors["old_duration_log_probs"]
                )
            joint_log_ratio = macro_policy_log_prob - old_macro_policy_log_prob
            joint_ratio = torch.exp(joint_log_ratio)
            joint_kl = torch.mean((joint_ratio - 1.0) - joint_log_ratio)
            joint_clip_fraction = torch.mean(
                (torch.abs(joint_ratio - 1.0) > clip).float()
            )
            value_loss = torch.mean(
                (components.value - tensors["macro_critic_targets"]) ** 2
            )
            duration_kl = torch.mean(
                (duration_ratio - 1.0) - duration_log_ratio
            )
            duration_clip_fraction = torch.mean(
                (torch.abs(duration_ratio - 1.0) > clip).float()
            )
            actor_loss = candidate_loss + duration_loss
            if self.unified_cache_action:
                entropy_mean = (components.cache_entropy + components.duration_entropy).mean()
            else:
                entropy_mean = (
                    components.conditional_update_entropy
                    + components.duration_entropy
                ).mean()
            loss = actor_loss - entropy_coefficient * entropy_mean
            self.macro_optimizer.zero_grad(set_to_none=True)
            if not self.unified_cache_action:
                self.macro_cache_trigger_actor_optimizer.zero_grad(set_to_none=True)
            self.macro_critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            macro_actor_parameters = self.policy.macro_actor_parameters()
            macro_critic_parameters = self.policy.macro_critic_parameters()
            trigger_parameters = self.policy.macro_cache_trigger_actor_parameters()
            if any(parameter.grad is not None for parameter in macro_critic_parameters):
                raise RuntimeError("Macro actor PPO update produced Macro Critic gradients.")
            if any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0.0).item())
                for parameter in trigger_parameters
            ):
                raise RuntimeError("Candidate/Duration loss produced Trigger Actor gradients.")
            macro_actor_before_clip = float(
                torch.nn.utils.clip_grad_norm_(macro_actor_parameters, max_grad_norm)
                .detach()
                .cpu()
            )
            macro_actor_after_clip = _gradient_norm(macro_actor_parameters)
            self.macro_optimizer.step()
            self.training_counters["macro_optimizer_steps"] += 1
            self.training_counters["macro_actor_optimizer_steps"] += 1
            records.append(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "candidate_actor_loss": float(candidate_loss.detach().cpu()),
                    "duration_actor_loss": float(duration_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_mean.detach().cpu()),
                    "approx_kl": float(joint_kl.detach().cpu()),
                    "clip_fraction": float(joint_clip_fraction.detach().cpu()),
                    "candidate_approx_kl": float(candidate_kl.detach().cpu()),
                    "candidate_clip_fraction": float(candidate_clip_fraction.detach().cpu()),
                    "duration_approx_kl": float(duration_kl.detach().cpu()),
                    "duration_clip_fraction": float(duration_clip_fraction.detach().cpu()),
                    "actor_grad_norm_before_clip": macro_actor_before_clip,
                    "actor_grad_norm_after_clip": macro_actor_after_clip,
                    "critic_grad_norm_before_clip": 0.0,
                    "critic_grad_norm_after_clip": 0.0,
                }
            )
            if target_kl > 0.0 and records[-1]["approx_kl"] > 1.5 * target_kl:
                stopped = True
                break

        trigger_records: list[dict[str, float]] = []
        trigger_stopped = False
        trigger_parameters = self.policy.macro_cache_trigger_actor_parameters()
        trigger_entropy_before = 0.0
        trigger_p_update_before = 0.0
        trigger_p_update_after = 0.0
        trigger_p_update_std_after = 0.0
        trigger_entropy_after = 0.0
        trigger_kl_mean = 0.0
        trigger_kl_max = 0.0
        trigger_approx_kl = 0.0
        trigger_clip_fraction = 0.0
        if len(eligible_trigger_indices):
            old_eligible_probabilities = np.stack(
                (
                    batch["old_trigger_p_keep"][eligible_trigger_indices],
                    batch["old_trigger_p_update"][eligible_trigger_indices],
                ),
                axis=-1,
            ).astype(np.float32)
            trigger_entropy_before = float(
                np.mean(
                    -np.sum(
                        np.where(
                            old_eligible_probabilities > 0.0,
                            old_eligible_probabilities
                            * np.log(np.maximum(old_eligible_probabilities, 1.0e-30)),
                            0.0,
                        ),
                        axis=-1,
                    )
                )
            )
            trigger_p_update_before = float(
                np.mean(old_eligible_probabilities[:, 1])
            )
        if len(active_trigger_indices):
            trigger_observations = torch.as_tensor(
                batch["observations"][active_trigger_indices],
                dtype=torch.float32,
                device=self.device,
            )
            trigger_actions = torch.as_tensor(
                batch["trigger_actions"][active_trigger_indices],
                dtype=torch.long,
                device=self.device,
            )
            trigger_old_log_probs = torch.as_tensor(
                batch["old_trigger_log_probs"][active_trigger_indices],
                dtype=torch.float32,
                device=self.device,
            )
            trigger_advantages = torch.as_tensor(
                trigger_scaled_advantages, dtype=torch.float32, device=self.device
            )
            has_update_candidates = torch.as_tensor(
                batch["candidate_masks"][active_trigger_indices, 1:].any(axis=-1),
                dtype=torch.bool,
                device=self.device,
            )
            if not bool(torch.all(has_update_candidates).item()):
                raise RuntimeError("Active Trigger PPO sample has no legal UPDATE candidate.")
            for _epoch in range(int(self.config["update_epochs"])):
                trigger_distribution = self.policy.cache_trigger_distribution(
                    trigger_observations, has_update_candidates
                )
                new_trigger_log_probs = trigger_distribution.log_prob(trigger_actions)
                log_ratio = new_trigger_log_probs - trigger_old_log_probs
                ratio = torch.exp(log_ratio)
                if not bool(torch.isfinite(ratio).all().item()):
                    raise FloatingPointError("Trigger PPO ratio contains NaN/Inf.")
                surrogate_loss = _clipped_ppo_surrogate_loss(
                    ratio, trigger_advantages, clip
                )
                trigger_entropy = trigger_distribution.entropy().mean()
                trigger_loss = surrogate_loss - entropy_coefficient * trigger_entropy
                if not bool(torch.isfinite(trigger_loss).item()):
                    raise FloatingPointError("Trigger PPO loss contains NaN/Inf.")
                approximate_kl = torch.mean((ratio - 1.0) - log_ratio)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > clip).float()
                )
                self.macro_cache_trigger_actor_optimizer.zero_grad(set_to_none=True)
                self.macro_optimizer.zero_grad(set_to_none=True)
                self.macro_critic_optimizer.zero_grad(set_to_none=True)
                trigger_loss.backward()
                if any(
                    parameter.grad is not None
                    for parameter in (
                        *self.policy.macro_actor_parameters(),
                        *self.policy.macro_critic_parameters(),
                    )
                ):
                    raise RuntimeError("Independent Trigger loss reached Candidate/Duration or Critic parameters.")
                trigger_grad_norm_before = float(
                    torch.nn.utils.clip_grad_norm_(trigger_parameters, max_grad_norm)
                    .detach()
                    .cpu()
                )
                trigger_grad_norm_after = _gradient_norm(trigger_parameters)
                self.macro_cache_trigger_actor_optimizer.step()
                self.training_counters["macro_cache_trigger_optimizer_steps"] += 1
                trigger_records.append(
                    {
                        "loss": float(trigger_loss.detach().cpu()),
                        "surrogate_loss": float(surrogate_loss.detach().cpu()),
                        "entropy": float(trigger_entropy.detach().cpu()),
                        "approx_kl": float(approximate_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "grad_norm_before_clip": trigger_grad_norm_before,
                        "grad_norm_after_clip": trigger_grad_norm_after,
                    }
                )
                if target_kl > 0.0 and trigger_records[-1]["approx_kl"] > 1.5 * target_kl:
                    trigger_stopped = True
                    break
        if len(eligible_trigger_indices):
            eligible_observations = torch.as_tensor(
                batch["observations"][eligible_trigger_indices],
                dtype=torch.float32,
                device=self.device,
            )
            eligible_has_update = torch.as_tensor(
                batch["candidate_masks"][eligible_trigger_indices, 1:].any(axis=-1),
                dtype=torch.bool,
                device=self.device,
            )
            old_probabilities = torch.as_tensor(
                np.stack(
                    (
                        batch["old_trigger_p_keep"][eligible_trigger_indices],
                        batch["old_trigger_p_update"][eligible_trigger_indices],
                    ),
                    axis=-1,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                final_distribution = self.policy.cache_trigger_distribution(
                    eligible_observations, eligible_has_update
                )
                final_probabilities = final_distribution.probs
                per_state_kl = torch.sum(
                    old_probabilities
                    * (
                        torch.log(old_probabilities.clamp_min(1.0e-30))
                        - final_distribution.logits
                    ),
                    dim=-1,
                ).clamp_min(0.0)
                trigger_kl_mean = float(per_state_kl.mean().cpu())
                trigger_kl_max = float(per_state_kl.max().cpu())
                trigger_entropy_after = float(final_distribution.entropy().mean().cpu())
                trigger_p_update_after = float(final_probabilities[:, 1].mean().cpu())
                trigger_p_update_std_after = float(
                    final_probabilities[:, 1].std(unbiased=False).cpu()
                )
            if len(active_trigger_indices):
                active_positions = np.searchsorted(
                    eligible_trigger_indices, active_trigger_indices
                )
                active_final_probs = final_probabilities[
                    torch.as_tensor(active_positions, dtype=torch.long, device=self.device)
                ]
                active_actions = torch.as_tensor(
                    batch["trigger_actions"][active_trigger_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                active_old_log_probs = torch.as_tensor(
                    batch["old_trigger_log_probs"][active_trigger_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                active_new_log_probs = torch.log(
                    active_final_probs.gather(1, active_actions[:, None]).squeeze(1)
                    .clamp_min(1.0e-30)
                )
                active_log_ratio = active_new_log_probs - active_old_log_probs
                active_ratio = torch.exp(active_log_ratio)
                if not bool(torch.isfinite(active_ratio).all().item()):
                    raise FloatingPointError("Post-update Trigger ratio contains NaN/Inf.")
                trigger_approx_kl = float(
                    torch.mean((active_ratio - 1.0) - active_log_ratio).cpu()
                )
                trigger_clip_fraction = float(
                    torch.mean((torch.abs(active_ratio - 1.0) > clip).float()).cpu()
                )
        self.training_counters["macro_cache_trigger_optimizer_steps"] += 0
        self.training_counters["macro_update_count"] += 1
        result = self._aggregate_update_metrics("macro", records)
        self.last1_macro_transition_buffer = {
            "episode_index": episode_index,
            "observations": current_observations.copy(),
            "next_observations": batch["next_observations"].astype(
                np.float32, copy=True
            ),
            "macro_rewards": batch["rewards"].astype(np.float32, copy=True),
            "actual_durations": batch["durations"].astype(np.int64, copy=True),
            "dones": batch["dones"].astype(bool, copy=True),
        }
        long_value_fit = self._update_long_value_critic_only(
            batch["observations"],
            realized_return_to_go,
            steps=self.long_value_steps_per_update,
            max_grad_norm=self.long_value_max_grad_norm,
        )
        current_pre = critic_fit["current_snapshots"]["0"]
        current_post = critic_fit["current_snapshots"][str(critic_steps)]
        last1_pre = (
            None
            if critic_fit["last1_snapshots"] is None
            else critic_fit["last1_snapshots"]["0"]
        )
        last1_post = (
            None
            if critic_fit["last1_snapshots"] is None
            else critic_fit["last1_snapshots"][str(critic_steps)]
        )
        with torch.no_grad():
            predictions = (
                self.policy.macro_value(
                    torch.as_tensor(
                        batch["observations"], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
        forced_keep_no_update_count = sum(
            not bool(np.any(item.cache_candidate_masks[1:]))
            for item in buffer.transitions
        )
        result.update(
            {
                "macro_actor_optimizer_steps_this_update": len(records),
                "macro_cache_trigger_optimizer_steps_this_update": len(trigger_records),
                "macro_critic_optimizer_steps_this_update": critic_fit["optimizer_steps"],
                "macro_critic_update_iterations_this_update": critic_fit[
                    "update_iterations"
                ],
                "long_value_optimizer_steps_this_update": long_value_fit[
                    "optimizer_steps"
                ],
                "macro_actor_advantage_mode": MACRO_ACTOR_ADVANTAGE_MODE,
                "macro_critic_target_mode": self.macro_critic_target_mode,
                "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
                "macro_actor_advantage_mean": float(adv_mean),
                "macro_actor_advantage_std": float(adv_std),
                "macro_candidate_advantage_mean": float(
                    np.mean(normalized_actor_advantage[selected_groups])
                    if np.any(selected_groups)
                    else 0.0
                ),
                "macro_candidate_advantage_std": float(
                    np.std(normalized_actor_advantage[selected_groups])
                    if np.any(selected_groups)
                    else 0.0
                ),
                "macro_duration_advantage_mean": float(np.mean(normalized_actor_advantage)),
                "macro_duration_advantage_std": float(np.std(normalized_actor_advantage)),
                "macro_candidate_conditional_approx_kl": float(
                    result["macro_candidate_approx_kl"]
                ),
                "macro_candidate_conditional_clip_fraction": float(
                    result["macro_candidate_clip_fraction"]
                ),
                "macro_duration_approx_kl": float(result["macro_duration_approx_kl"]),
                "macro_duration_clip_fraction": float(result["macro_duration_clip_fraction"]),
                "macro_trigger_p_keep_mean": float(np.mean(old_p_keep)),
                "macro_trigger_p_update_mean": float(np.mean(old_p_update)),
                "trigger_p_keep_mean": float(np.mean(old_p_keep)),
                "trigger_p_update_mean": float(np.mean(old_p_update)),
                "trigger_credit_source": "paired critic-independent option reward difference",
                "trigger_critic_bootstrap_used": False,
                "trigger_candidate_sample_count": TRIGGER_CANDIDATE_SAMPLES,
                "trigger_uncertainty_gate_active": True,
                "trigger_teacher_valid_count": int(np.count_nonzero(batch["trigger_teacher_valid"])),
                "trigger_credit_eligible_count": int(len(eligible_trigger_indices)),
                "trigger_credit_active_count": int(len(active_trigger_indices)),
                "trigger_credit_inactive_uncertain_count": int(
                    len(eligible_trigger_indices) - len(active_trigger_indices)
                ),
                "trigger_credit_active_fraction": trigger_active_fraction,
                "trigger_advantage_raw_mean": (
                    float(np.mean(trigger_raw_advantages)) if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_advantage_raw_std": (
                    float(np.std(trigger_raw_advantages)) if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_advantage_raw_median": (
                    float(np.median(trigger_raw_advantages)) if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_advantage_raw_mad": trigger_raw_mad,
                "trigger_advantage_active_raw_mean": (
                    float(np.mean(active_trigger_raw_advantages))
                    if len(active_trigger_raw_advantages) else 0.0
                ),
                "trigger_advantage_active_raw_std": (
                    float(np.std(active_trigger_raw_advantages))
                    if len(active_trigger_raw_advantages) else 0.0
                ),
                "trigger_advantage_scale": trigger_advantage_scale,
                "trigger_advantage_scaled_mean": (
                    float(np.mean(trigger_scaled_advantages))
                    if len(trigger_scaled_advantages) else 0.0
                ),
                "trigger_advantage_scaled_std": (
                    float(np.std(trigger_scaled_advantages))
                    if len(trigger_scaled_advantages) else 0.0
                ),
                "trigger_advantage_scaled_min": (
                    float(np.min(trigger_scaled_advantages))
                    if len(trigger_scaled_advantages) else 0.0
                ),
                "trigger_advantage_scaled_max": (
                    float(np.max(trigger_scaled_advantages))
                    if len(trigger_scaled_advantages) else 0.0
                ),
                "trigger_positive_advantage_fraction": (
                    float(np.mean(active_trigger_raw_advantages > 0.0))
                    if len(active_trigger_raw_advantages) else 0.0
                ),
                "trigger_negative_advantage_fraction": (
                    float(np.mean(active_trigger_raw_advantages < 0.0))
                    if len(active_trigger_raw_advantages) else 0.0
                ),
                "trigger_confidence_ratio_median": (
                    float(np.median(trigger_confidence_ratios))
                    if len(trigger_confidence_ratios) else 0.0
                ),
                "trigger_confidence_ratio_p10": (
                    float(np.percentile(trigger_confidence_ratios, 10.0))
                    if len(trigger_confidence_ratios) else 0.0
                ),
                "trigger_confidence_ratio_p90": (
                    float(np.percentile(trigger_confidence_ratios, 90.0))
                    if len(trigger_confidence_ratios) else 0.0
                ),
                "trigger_entropy_before": trigger_entropy_before,
                "trigger_entropy": trigger_entropy_after,
                "trigger_p_update_before": trigger_p_update_before,
                "trigger_p_update_after": trigger_p_update_after,
                "trigger_p_keep_after": 1.0 - trigger_p_update_after,
                "trigger_p_update_std_after": trigger_p_update_std_after,
                "trigger_kl_mean": trigger_kl_mean,
                "trigger_kl_max": trigger_kl_max,
                "trigger_approx_kl": trigger_approx_kl,
                "trigger_clip_fraction": trigger_clip_fraction,
                "trigger_ppo_epoch_count": int(len(trigger_records)),
                "trigger_ppo_early_stop": bool(trigger_stopped),
                "trigger_ppo_active": True,
                "trigger_teacher_q_keep_mean": float(
                    np.mean(batch["trigger_teacher_q_keep"][eligible_trigger_indices])
                    if len(eligible_trigger_indices) else 0.0
                ),
                "trigger_teacher_q_update_mean": float(
                    np.mean(batch["trigger_teacher_q_update"][eligible_trigger_indices])
                    if len(eligible_trigger_indices) else 0.0
                ),
                "trigger_teacher_delta_mean": float(
                    np.mean(trigger_raw_advantages) if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_teacher_delta_median": (
                    float(np.median(trigger_raw_advantages))
                    if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_teacher_delta_std": (
                    float(np.std(trigger_raw_advantages))
                    if len(trigger_raw_advantages) else 0.0
                ),
                "trigger_update_standard_error_mean": float(
                    np.mean(batch["trigger_update_standard_error"][eligible_trigger_indices])
                    if len(eligible_trigger_indices) else 0.0
                ),
                "trigger_update_standard_error_median": float(
                    np.median(batch["trigger_update_standard_error"][eligible_trigger_indices])
                    if len(eligible_trigger_indices) else 0.0
                ),
                    "joint_logprob_decomposition_max_error": max_decomposition_error,
                    "rollout_logprob_max_error": rollout_logprob_max_error,
                "macro_early_stop": bool(stopped),
                "macro_critic_target_mean": float(np.mean(macro_critic_target)),
                "macro_critic_target_std": float(np.std(macro_critic_target)),
                "macro_critic_target_min": float(np.min(macro_critic_target)),
                "macro_critic_target_max": float(np.max(macro_critic_target)),
                "old_value_mean": float(np.mean(batch["old_values"])),
                "old_value_std": float(np.std(batch["old_values"])),
                "old_next_value_mean": float(np.mean(next_values)),
                "old_next_value_std": float(np.std(next_values)),
                "td_error_mean": float(np.mean(macro_critic_target - batch["old_values"])),
                "td_error_std": float(np.std(macro_critic_target - batch["old_values"])),
                "realized_return_to_go_mean": float(np.mean(realized_return_to_go)),
                "realized_return_to_go_std": float(np.std(realized_return_to_go)),
                "critic_target_detached": critic_fit["target_detached"],
                "critic_target_frozen_during_update": critic_fit["target_frozen"],
                "critic_target_sha256_before": critic_fit["current_target_sha256_before"],
                "critic_target_sha256_after": critic_fit["current_target_sha256_after"],
                "current_and_last1_use_same_frozen_value_snapshot": True,
                "current_target_frozen": critic_fit["target_frozen"],
                "last1_target_frozen": critic_fit["last1_target_frozen"],
                "macro_critic_common_n2_target": True,
                "macro_critic_target_detached": critic_fit["target_detached"],
                "macro_critic_target_frozen_during_50_steps": critic_fit[
                    "target_frozen"
                ],
                "macro_adv_mean": float(adv_mean),
                "macro_adv_std": float(adv_std),
                "macro_normalized_adv_mean": float(np.mean(normalized_actor_advantage)),
                "macro_normalized_adv_std": float(np.std(normalized_actor_advantage)),
                "macro_value_prediction_mean": float(np.mean(predictions)),
                "macro_value_prediction_std": float(np.std(predictions)),
                "macro_value_target_mean": float(np.mean(macro_critic_target)),
                "macro_value_target_std": float(np.std(macro_critic_target)),
                "macro_value_target_prediction_corr": _pearson_correlation(
                    macro_critic_target, predictions
                ),
                "macro_value_rmse": float(
                    np.sqrt(np.mean((macro_critic_target - predictions) ** 2))
                ),
                "macro_explained_variance": current_post["explained_variance"],
                "pre_update_target_ev": current_pre["explained_variance"],
                "post_update_target_ev": current_post["explained_variance"],
                "pre_update_target_correlation": current_pre["correlation"],
                "post_update_target_correlation": current_post["correlation"],
                "pre_update_target_std_ratio": current_pre["std_ratio"],
                "post_update_target_std_ratio": current_post["std_ratio"],
                "pre_update_return_ev": None,
                "post_update_return_ev": None,
                "macro_critic_fit_snapshots": critic_fit["fit_snapshots"],
                "n2_pre_ev": current_pre["explained_variance"],
                "n2_post_ev": current_post["explained_variance"],
                "n2_pre_correlation": current_pre["correlation"],
                "n2_post_correlation": current_post["correlation"],
                "n2_pre_std_ratio": current_pre["std_ratio"],
                "n2_post_std_ratio": current_post["std_ratio"],
                "n2_std_ratio": current_post["std_ratio"],
                "current_pre_ev": current_pre["explained_variance"],
                "current_post_ev": current_post["explained_variance"],
                "last1_pre_update_ev": (
                    None if last1_pre is None else last1_pre["explained_variance"]
                ),
                "last1_post_update_ev": (
                    None if last1_post is None else last1_post["explained_variance"]
                ),
                "retention_delta": (
                    None
                    if last1_pre is None
                    or last1_post is None
                    or last1_pre["explained_variance"] is None
                    or last1_post["explained_variance"] is None
                    else float(
                        last1_post["explained_variance"]
                        - last1_pre["explained_variance"]
                    )
                ),
                "last1_policy_age_updates": None if last1_buffer is None else 1,
                "target_std_current": critic_fit["target_std_current"],
                "target_std_last1": critic_fit["target_std_last1"],
                "target_std_used_for_gradient_scaling": False,
                "target_scale_normalization_active": False,
                "raw_mse_current": critic_fit["raw_mse_current_mean"],
                "raw_mse_last1": critic_fit["raw_mse_last1_mean"],
                "episode1_v10_compatibility_path": episode_index == 1,
                "critic_group_episode_indices": (
                    [episode_index]
                    if last1_buffer is None
                    else [episode_index, int(last1_buffer["episode_index"])]
                ),
                "macro_critic_grad_norm_before_clip": critic_fit[
                    "combined_grad_norm_before_clip_mean"
                ],
                "macro_critic_grad_norm_before_clip_p95": critic_fit[
                    "combined_grad_norm_before_clip_p95"
                ],
                "macro_critic_grad_norm_before_clip_max": critic_fit[
                    "combined_grad_norm_before_clip_max"
                ],
                "macro_critic_grad_clip_fraction": critic_fit[
                    "grad_clip_fraction"
                ],
                "macro_critic_grad_norm_after_clip": critic_fit[
                    "combined_grad_norm_after_clip_mean"
                ],
                "raw_grad_cosine_current_last1": critic_fit[
                    "raw_grad_cosine_by_step"
                ],
                "grad_cosine_raw_mean": critic_fit["grad_cosine_raw_mean"],
                "grad_cosine_raw_median": critic_fit["grad_cosine_raw_median"],
                "grad_cosine_raw_min": critic_fit["grad_cosine_raw_min"],
                "grad_cosine_raw_p10": critic_fit["grad_cosine_raw_p10"],
                "grad_conflict_fraction": critic_fit["grad_conflict_fraction"],
                "post_projection_cosine_mean": critic_fit[
                    "post_projection_cosine_mean"
                ],
                "post_projection_cosine_min": critic_fit[
                    "post_projection_cosine_min"
                ],
                "post_projection_cosine_max": critic_fit[
                    "post_projection_cosine_max"
                ],
                "post_projection_cosine_max_abs_on_conflict": critic_fit[
                    "post_projection_cosine_max_abs_on_conflict"
                ],
                "projection_orthogonality_max_abs_error": critic_fit[
                    "projection_orthogonality_max_abs_error"
                ],
                "pcgrad_projection_active_fraction": critic_fit[
                    "pcgrad_projection_active_fraction"
                ],
                "pcgrad_priority": "CURRENT",
                "current_gradient_projected": False,
                "current_grad_norm_raw_mean": critic_fit[
                    "current_grad_norm_raw_mean"
                ],
                "current_grad_norm_raw_p95": critic_fit[
                    "current_grad_norm_raw_p95"
                ],
                "current_grad_norm_raw_max": critic_fit[
                    "current_grad_norm_raw_max"
                ],
                "last1_grad_norm_raw_mean": critic_fit[
                    "last1_grad_norm_raw_mean"
                ],
                "last1_grad_norm_raw_p95": critic_fit[
                    "last1_grad_norm_raw_p95"
                ],
                "last1_grad_norm_raw_max": critic_fit[
                    "last1_grad_norm_raw_max"
                ],
                "last1_grad_norm_scaled_mean": critic_fit[
                    "last1_grad_norm_scaled_mean"
                ],
                "last1_grad_norm_scaled_p95": critic_fit[
                    "last1_grad_norm_scaled_p95"
                ],
                "last1_grad_norm_scaled_max": critic_fit[
                    "last1_grad_norm_scaled_max"
                ],
                "last1_norm_scale_factor_mean": critic_fit[
                    "last1_norm_scale_factor_mean"
                ],
                "scaled_to_current_norm_ratio_mean": critic_fit[
                    "scaled_to_current_norm_ratio_mean"
                ],
                "scaled_to_current_norm_ratio_max_abs_error": critic_fit[
                    "scaled_to_current_norm_ratio_max_abs_error"
                ],
                "current_grad_zero_fallback_fraction": critic_fit[
                    "current_grad_zero_fallback_fraction"
                ],
                "grad_norm_mix_before_final_rescale_mean": critic_fit[
                    "grad_norm_mix_before_final_rescale_mean"
                ],
                "grad_norm_final_before_clip_mean": critic_fit[
                    "grad_norm_final_before_clip_mean"
                ],
                "grad_norm_final_before_clip_p95": critic_fit[
                    "grad_norm_final_before_clip_p95"
                ],
                "grad_norm_final_before_clip_max": critic_fit[
                    "grad_norm_final_before_clip_max"
                ],
                "final_to_current_norm_ratio_mean": critic_fit[
                    "final_to_current_norm_ratio_mean"
                ],
                "final_to_current_norm_ratio_max_abs_error": critic_fit[
                    "final_to_current_norm_ratio_max_abs_error"
                ],
                "combined_grad_norm_before_clip_mean": critic_fit[
                    "combined_grad_norm_before_clip_mean"
                ],
                "combined_grad_norm_before_clip_p95": critic_fit[
                    "combined_grad_norm_before_clip_p95"
                ],
                "combined_grad_norm_before_clip_max": critic_fit[
                    "combined_grad_norm_before_clip_max"
                ],
                "combined_grad_norm_after_clip_mean": critic_fit[
                    "combined_grad_norm_after_clip_mean"
                ],
                "current_to_last1_grad_norm_ratio": critic_fit[
                    "current_to_last1_grad_norm_ratio"
                ],
                "macro_actor_loss_changed": True,
                "smdp_gae_changed": False,
                "actor_uses_last1_data": False,
                "macro_critic_value_loss_coefficient": value_loss_coefficient,
                "macro_critic_value_loss": current_post["mse"],
                "macro_early_stop": bool(stopped),
                "long_value_target": self.long_value_target,
                "long_value_fit_snapshots": long_value_fit["snapshots"],
                "long_value_target_detached": long_value_fit["target_detached"],
                "long_value_target_frozen_during_update": long_value_fit[
                    "target_frozen"
                ],
                "long_value_target_sha256_before": long_value_fit[
                    "target_sha256_before"
                ],
                "long_value_target_sha256_after": long_value_fit[
                    "target_sha256_after"
                ],
                "long_value_pre_ev": long_value_fit["snapshots"]["0"][
                    "explained_variance"
                ],
                "long_value_post_ev": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["explained_variance"],
                "long_value_pre_correlation": long_value_fit["snapshots"]["0"][
                    "correlation"
                ],
                "long_value_post_correlation": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["correlation"],
                "long_value_pre_std_ratio": long_value_fit["snapshots"]["0"][
                    "std_ratio"
                ],
                "long_value_post_std_ratio": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["std_ratio"],
                "long_value_rmse": long_value_fit["snapshots"][
                    str(self.long_value_steps_per_update)
                ]["rmse"],
            }
        )
        if self.unified_cache_action:
            result.update(
                {
                    "unified_cache_action_active": True,
                    "trigger_active": False,
                    "trigger_ppo_active": False,
                    "trigger_optimizer_steps_this_update": 0,
                    "macro_cache_trigger_optimizer_steps_this_update": 0,
                    "trigger_credit_source": "inactive",
                    "trigger_candidate_sample_count": 0,
                    "trigger_uncertainty_gate_active": False,
                    "trigger_teacher_used": False,
                    "trigger_counterfactual_branches": 0,
                    "macro_actor_advantage_mode": "unified_cache_candidate_macro_smdp_gae_ppo",
                    "macro_candidate_advantage_mean": float(np.mean(normalized_actor_advantage)),
                    "macro_candidate_advantage_std": float(np.std(normalized_actor_advantage)),
                    "macro_candidate_conditional_approx_kl": float(
                        result["macro_candidate_approx_kl"]
                    ),
                    "macro_candidate_conditional_clip_fraction": float(
                        result["macro_candidate_clip_fraction"]
                    ),
                }
            )
        return result

    def _update_macro_critic_only(
        self,
        current_observations: np.ndarray,
        current_targets: np.ndarray,
        *,
        last1_observations: np.ndarray | None,
        last1_targets: np.ndarray | None,
        steps: int,
        value_loss_coefficient: float,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """Fit raw-MSE N2 groups with V10 current-gradient scale preserved."""

        if int(steps) != 50:
            raise ValueError("Macro Critic update budget is fixed at 50 iterations.")
        if not math.isclose(float(value_loss_coefficient), 0.01, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("V14b requires the V10 Macro Critic loss coefficient 0.01.")
        if (last1_observations is None) != (last1_targets is None):
            raise ValueError("last1 observations and targets must be provided together.")
        current_observation_tensor = torch.as_tensor(
            current_observations, dtype=torch.float32, device=self.device
        ).detach()
        current_target_tensor = torch.as_tensor(
            current_targets, dtype=torch.float32, device=self.device
        ).reshape(-1).detach().clone()
        if (
            current_observation_tensor.ndim != 2
            or int(current_observation_tensor.shape[-1]) != self.policy.observation_dim
            or len(current_observation_tensor) != len(current_target_tensor)
            or len(current_target_tensor) == 0
        ):
            raise ValueError("Current Macro Critic observations/targets shape mismatch.")
        last1_observation_tensor = None
        last1_target_tensor = None
        if last1_observations is not None:
            last1_observation_tensor = torch.as_tensor(
                last1_observations, dtype=torch.float32, device=self.device
            ).detach()
            last1_target_tensor = torch.as_tensor(
                last1_targets, dtype=torch.float32, device=self.device
            ).reshape(-1).detach().clone()
            if (
                last1_observation_tensor.ndim != 2
                or int(last1_observation_tensor.shape[-1]) != self.policy.observation_dim
                or len(last1_observation_tensor) != len(last1_target_tensor)
                or len(last1_target_tensor) == 0
            ):
                raise ValueError("Last1 Macro Critic observations/targets shape mismatch.")
        tensors_to_check = [current_observation_tensor, current_target_tensor]
        if last1_observation_tensor is not None:
            tensors_to_check.extend([last1_observation_tensor, last1_target_tensor])
        if not all(bool(torch.all(torch.isfinite(item)).item()) for item in tensors_to_check):
            raise FloatingPointError("Macro Critic group contains NaN/Inf.")
        frozen_current_targets = current_target_tensor.clone()
        frozen_last1_targets = (
            None if last1_target_tensor is None else last1_target_tensor.clone()
        )
        current_target_std = float(current_target_tensor.std(unbiased=False).cpu())
        last1_target_std = (
            None
            if last1_target_tensor is None
            else float(last1_target_tensor.std(unbiased=False).cpu())
        )
        parameters = tuple(self.policy.macro_critic_parameters())
        actor_parameters = self.policy.macro_actor_parameters()
        self.macro_optimizer.zero_grad(set_to_none=True)
        self.micro_optimizer.zero_grad(set_to_none=True)
        self.macro_critic_optimizer.zero_grad(set_to_none=True)
        current_sha_before = hashlib.sha256(
            current_target_tensor.cpu().numpy().tobytes()
        ).hexdigest()
        last1_sha_before = (
            None
            if last1_target_tensor is None
            else hashlib.sha256(last1_target_tensor.cpu().numpy().tobytes()).hexdigest()
        )

        def fit_snapshot(
            observations: Any, targets: Any
        ) -> dict[str, float | None]:
            return self._macro_critic_fit_metrics(observations, targets)

        current_snapshots: dict[str, dict[str, float | None]] = {
            "0": fit_snapshot(current_observation_tensor, current_target_tensor)
        }
        last1_snapshots: dict[str, dict[str, float | None]] | None = (
            None
            if last1_observation_tensor is None
            else {"0": fit_snapshot(last1_observation_tensor, last1_target_tensor)}
        )
        fit_snapshots: dict[str, dict[str, Any]] = {
            "0": {"current": current_snapshots["0"], "last1": None if last1_snapshots is None else last1_snapshots["0"]}
        }
        raw_mse_current: list[float] = []
        raw_mse_last1: list[float] = []
        cosine_raw_values: list[float] = []
        cosine_post_values: list[float] = []
        post_cosine_abs_on_conflict: list[float] = []
        current_gradient_norms: list[float] = []
        last1_raw_gradient_norms: list[float] = []
        last1_scaled_gradient_norms: list[float] = []
        mixed_gradient_norms: list[float] = []
        final_gradient_norms: list[float] = []
        combined_gradient_norms_after_clip: list[float] = []
        gradient_clip_events: list[bool] = []
        projection_events: list[bool] = []
        last1_scale_factors: list[float] = []
        scaled_to_current_ratios: list[float] = []
        final_to_current_ratios: list[float] = []
        current_zero_fallback_events: list[bool] = []
        optimizer_steps = 0
        snapshot_steps = {10, 25, 50}
        for step in range(1, int(steps) + 1):
            current_prediction = self.policy.macro_value(current_observation_tensor)
            current_errors = current_prediction - current_target_tensor
            current_raw_mse_value = torch.mean(current_errors.square())
            current_loss = 0.01 * current_raw_mse_value
            if not bool(torch.isfinite(current_loss).item()):
                raise FloatingPointError("Current Macro Critic loss is non-finite.")
            raw_mse_current.append(float(current_raw_mse_value.detach().cpu()))

            if last1_observation_tensor is not None:
                assert last1_target_tensor is not None
                current_gradients_optional = torch.autograd.grad(
                    current_loss, parameters, allow_unused=True
                )
                current_gradients = tuple(
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                    for parameter, gradient in zip(
                        parameters, current_gradients_optional, strict=True
                    )
                )
                last1_prediction = self.policy.macro_value(last1_observation_tensor)
                last1_errors = last1_prediction - last1_target_tensor
                last1_raw_mse_value = torch.mean(last1_errors.square())
                last1_loss = 0.01 * last1_raw_mse_value
                if not bool(torch.isfinite(last1_loss).item()):
                    raise FloatingPointError("Last1 Macro Critic loss is non-finite.")
                last1_gradients_optional = torch.autograd.grad(
                    last1_loss, parameters, allow_unused=True
                )
                last1_gradients = tuple(
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                    for parameter, gradient in zip(
                        parameters, last1_gradients_optional, strict=True
                    )
                )
                pcgrad = _pcgrad_lite_current_priority(
                    tuple(current_gradients), tuple(last1_gradients)
                )
                gradients_to_apply = pcgrad["combined"]
                current_gradient_norms.append(pcgrad["current_norm"])
                last1_raw_gradient_norms.append(pcgrad["last1_norm_raw"])
                last1_scaled_gradient_norms.append(pcgrad["last1_norm_scaled"])
                mixed_gradient_norms.append(pcgrad["mixed_norm"])
                final_gradient_norms.append(pcgrad["combined_norm_before_clip"])
                cosine_raw_values.append(pcgrad["cosine_raw"])
                cosine_post_values.append(pcgrad["cosine_post_projection"])
                projection_events.append(pcgrad["projection_active"])
                if pcgrad["projection_active"]:
                    post_cosine_abs_on_conflict.append(
                        abs(pcgrad["cosine_post_projection"])
                    )
                last1_scale_factors.append(pcgrad["last1_scale_factor"])
                scaled_to_current_ratios.append(
                    pcgrad["scaled_to_current_norm_ratio"]
                )
                final_to_current_ratios.append(
                    pcgrad["final_to_current_norm_ratio"]
                )
                current_zero_fallback_events.append(pcgrad["current_zero_fallback"])
                raw_mse_last1.append(float(last1_raw_mse_value.detach().cpu()))
            else:
                # Episode 1 has no last1 group, so apply the raw current-loss gradient.
                self.macro_critic_optimizer.zero_grad(set_to_none=True)
                current_loss.backward()
                current_gradient_norm = _gradient_norm(parameters)
                current_gradient_norms.append(current_gradient_norm)
                final_gradient_norms.append(current_gradient_norm)
                mixed_gradient_norms.append(current_gradient_norm)
                gradients_to_apply = None
            if gradients_to_apply is not None:
                self.macro_critic_optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, gradients_to_apply, strict=True):
                    parameter.grad = gradient.detach().clone()
            if any(parameter.grad is not None for parameter in actor_parameters):
                raise RuntimeError("Macro Critic-only update produced Macro Actor gradients.")
            grad_before_clip = float(
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu()
            )
            if not math.isfinite(grad_before_clip):
                raise FloatingPointError("Macro Critic combined gradient norm is non-finite.")
            grad_after_clip = _gradient_norm(parameters)
            combined_gradient_norms_after_clip.append(grad_after_clip)
            gradient_clip_events.append(grad_before_clip > float(max_grad_norm))
            self.macro_critic_optimizer.step()
            optimizer_steps += 1
            self.training_counters["macro_optimizer_steps"] += 1
            self.training_counters["macro_critic_optimizer_steps"] += 1
            if not torch.equal(current_target_tensor, frozen_current_targets):
                raise RuntimeError("Current N2 target changed during Critic optimization.")
            if last1_target_tensor is not None and not torch.equal(
                last1_target_tensor, frozen_last1_targets
            ):
                raise RuntimeError("Last1 N2 target changed during Critic optimization.")
            if step in snapshot_steps:
                current_snapshots[str(step)] = fit_snapshot(
                    current_observation_tensor, current_target_tensor
                )
                last1_snap = (
                    None
                    if last1_observation_tensor is None
                    else fit_snapshot(last1_observation_tensor, last1_target_tensor)
                )
                if last1_snap is not None:
                    assert last1_snapshots is not None
                    last1_snapshots[str(step)] = last1_snap
                fit_snapshots[str(step)] = {
                    "current": current_snapshots[str(step)],
                    "last1": last1_snap,
                }
        current_sha_after = hashlib.sha256(
            current_target_tensor.cpu().numpy().tobytes()
        ).hexdigest()
        last1_sha_after = (
            None
            if last1_target_tensor is None
            else hashlib.sha256(last1_target_tensor.cpu().numpy().tobytes()).hexdigest()
        )
        target_frozen = bool(
            torch.equal(current_target_tensor, frozen_current_targets)
            and current_sha_before == current_sha_after
        )
        last1_target_frozen = (
            None
            if last1_target_tensor is None
            else bool(
                torch.equal(last1_target_tensor, frozen_last1_targets)
                and last1_sha_before == last1_sha_after
            )
        )
        if not target_frozen or last1_target_frozen is False:
            raise RuntimeError("V14 N2 group targets must stay detached and frozen.")
        if current_target_tensor.requires_grad or (
            last1_target_tensor is not None and last1_target_tensor.requires_grad
        ):
            raise RuntimeError("V14 N2 group targets must be detached.")
        if optimizer_steps != int(steps):
            raise RuntimeError("V14 scalar Macro Critic optimizer step count mismatch.")

        def summarize(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "p95": None, "max": None, "median": None, "min": None, "p10": None}
            return {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "p10": float(np.percentile(values, 10)),
            }

        current_norm_stats = summarize(current_gradient_norms)
        last1_raw_norm_stats = summarize(last1_raw_gradient_norms)
        last1_scaled_norm_stats = summarize(last1_scaled_gradient_norms)
        mixed_norm_stats = summarize(mixed_gradient_norms)
        final_norm_stats = summarize(final_gradient_norms)
        clipped_norm_stats = summarize(combined_gradient_norms_after_clip)
        cosine_stats = summarize(cosine_raw_values)
        post_cosine_stats = summarize(cosine_post_values)
        last1_scale_stats = summarize(last1_scale_factors)
        scaled_ratio_stats = summarize(scaled_to_current_ratios)
        final_ratio_stats = summarize(final_to_current_ratios)
        return {
            "optimizer_steps": int(optimizer_steps),
            "update_iterations": int(steps),
            "target_detached": True,
            "target_frozen": target_frozen,
            "last1_target_frozen": last1_target_frozen,
            "current_target_sha256_before": current_sha_before,
            "current_target_sha256_after": current_sha_after,
            "last1_target_sha256_before": last1_sha_before,
            "last1_target_sha256_after": last1_sha_after,
            "current_snapshots": current_snapshots,
            "last1_snapshots": last1_snapshots,
            "fit_snapshots": fit_snapshots,
            "target_std_current": current_target_std,
            "target_std_last1": last1_target_std,
            "raw_mse_current_mean": float(np.mean(raw_mse_current)),
            "raw_mse_last1_mean": None if not raw_mse_last1 else float(np.mean(raw_mse_last1)),
            "raw_grad_cosine_by_step": cosine_raw_values,
            "grad_cosine_raw_mean": cosine_stats["mean"],
            "grad_cosine_raw_median": cosine_stats["median"],
            "grad_cosine_raw_min": cosine_stats["min"],
            "grad_cosine_raw_p10": cosine_stats["p10"],
            "grad_conflict_fraction": (
                None if not cosine_raw_values else float(np.mean(np.asarray(cosine_raw_values) < 0.0))
            ),
            "post_projection_cosine_by_step": cosine_post_values,
            "post_projection_cosine_mean": post_cosine_stats["mean"],
            "post_projection_cosine_min": post_cosine_stats["min"],
            "post_projection_cosine_max": post_cosine_stats["max"],
            "post_projection_cosine_max_abs_on_conflict": (
                None
                if not post_cosine_abs_on_conflict
                else float(max(post_cosine_abs_on_conflict))
            ),
            "projection_orthogonality_max_abs_error": (
                None
                if not post_cosine_abs_on_conflict
                else float(max(post_cosine_abs_on_conflict))
            ),
            "pcgrad_projection_active_fraction": (
                None if not projection_events else float(np.mean(projection_events))
            ),
            "current_grad_zero_fallback_fraction": (
                None
                if not current_zero_fallback_events
                else float(np.mean(current_zero_fallback_events))
            ),
            "current_grad_norm_raw_mean": current_norm_stats["mean"],
            "current_grad_norm_raw_p95": current_norm_stats["p95"],
            "current_grad_norm_raw_max": current_norm_stats["max"],
            "last1_grad_norm_raw_mean": last1_raw_norm_stats["mean"],
            "last1_grad_norm_raw_p95": last1_raw_norm_stats["p95"],
            "last1_grad_norm_raw_max": last1_raw_norm_stats["max"],
            "last1_grad_norm_scaled_mean": last1_scaled_norm_stats["mean"],
            "last1_grad_norm_scaled_p95": last1_scaled_norm_stats["p95"],
            "last1_grad_norm_scaled_max": last1_scaled_norm_stats["max"],
            "last1_norm_scale_factor_mean": last1_scale_stats["mean"],
            "scaled_to_current_norm_ratio_mean": scaled_ratio_stats["mean"],
            "scaled_to_current_norm_ratio_max_abs_error": (
                None
                if not scaled_to_current_ratios
                else float(np.max(np.abs(np.asarray(scaled_to_current_ratios) - 1.0)))
            ),
            "grad_norm_mix_before_final_rescale_mean": mixed_norm_stats["mean"],
            "grad_norm_final_before_clip_mean": final_norm_stats["mean"],
            "grad_norm_final_before_clip_p95": final_norm_stats["p95"],
            "grad_norm_final_before_clip_max": final_norm_stats["max"],
            "combined_grad_norm_before_clip_mean": final_norm_stats["mean"],
            "combined_grad_norm_before_clip_p95": final_norm_stats["p95"],
            "combined_grad_norm_before_clip_max": final_norm_stats["max"],
            "combined_grad_norm_after_clip_mean": clipped_norm_stats["mean"],
            "grad_clip_fraction": float(np.mean(gradient_clip_events)),
            "final_to_current_norm_ratio_mean": final_ratio_stats["mean"],
            "final_to_current_norm_ratio_max_abs_error": (
                None
                if not final_to_current_ratios
                else float(np.max(np.abs(np.asarray(final_to_current_ratios) - 1.0)))
            ),
            "current_to_last1_grad_norm_ratio": scaled_ratio_stats["mean"],
        }

    def _macro_critic_fit_metrics(
        self, observations: Any, value_targets: Any
    ) -> dict[str, float | None]:
        with torch.no_grad():
            predictions = (
                self.policy.macro_value(observations)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        target_tensor = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)
        targets = target_tensor.detach().cpu().numpy().astype(np.float64)
        if predictions.shape != targets.shape:
            raise ValueError("Macro Critic prediction/target shape mismatch.")
        target_std = float(np.std(targets))
        prediction_std = float(np.std(predictions))
        rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
        return {
            "explained_variance": _explained_variance(targets, predictions),
            "rmse": rmse,
            "mse": float(rmse**2),
            "prediction_std": prediction_std,
            "target_std": target_std,
            "std_ratio": prediction_std / target_std if target_std > 1.0e-12 else None,
            "correlation": _pearson_correlation(targets, predictions),
        }

    def _update_long_value_critic_only(
        self,
        observations: np.ndarray,
        realized_return_to_go: np.ndarray,
        *,
        steps: int,
        max_grad_norm: float,
    ) -> dict[str, Any]:
        """Fit the independent Long Critic to one episode's frozen realized returns."""

        if int(steps) != 50:
            raise ValueError("Long-Value Critic update budget is fixed at 50 steps.")
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).detach()
        target_tensor = torch.as_tensor(
            realized_return_to_go, dtype=torch.float32, device=self.device
        ).reshape(-1).detach().clone()
        if (
            observation_tensor.ndim != 2
            or int(observation_tensor.shape[-1]) != int(self.policy.observation_dim)
            or int(observation_tensor.shape[0]) != int(target_tensor.shape[0])
            or int(target_tensor.numel()) == 0
        ):
            raise ValueError("Long-Value Critic episode observation/target shape mismatch.")
        if not bool(torch.all(torch.isfinite(observation_tensor)).item()) or not bool(
            torch.all(torch.isfinite(target_tensor)).item()
        ):
            raise FloatingPointError("Long-Value Critic batch contains NaN/Inf.")

        frozen_targets = target_tensor.clone()
        target_sha256_before = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        long_parameters = self.policy.long_value_critic_parameters()
        long_ids = {id(parameter) for parameter in long_parameters}
        macro_optimizer_ids = {
            id(parameter)
            for group in self.macro_optimizer.param_groups
            for parameter in group["params"]
        }
        macro_critic_optimizer_ids = {
            id(parameter)
            for group in self.macro_critic_optimizer.param_groups
            for parameter in group["params"]
        }
        micro_optimizer_ids = {
            id(parameter)
            for group in self.micro_optimizer.param_groups
            for parameter in group["params"]
        }
        long_optimizer_ids = {
            id(parameter)
            for group in self.long_value_optimizer.param_groups
            for parameter in group["params"]
        }
        if (
            long_optimizer_ids != long_ids
            or long_ids.intersection(macro_optimizer_ids)
            or long_ids.intersection(macro_critic_optimizer_ids)
            or long_ids.intersection(micro_optimizer_ids)
        ):
            raise RuntimeError(
                "Long-Value Critic optimizer parameters overlap with Actor/N2/Micro."
            )
        protected_parameters = tuple(
            self.policy.macro_actor_parameters()
            + self.policy.macro_critic_parameters()
            + self.policy.micro_actor_parameters()
            + self.policy.micro_critic_parameters()
        )
        protected_before = tuple(parameter.detach().clone() for parameter in protected_parameters)
        self.macro_optimizer.zero_grad(set_to_none=True)
        self.macro_critic_optimizer.zero_grad(set_to_none=True)
        self.micro_optimizer.zero_grad(set_to_none=True)
        self.long_value_optimizer.zero_grad(set_to_none=True)

        def snapshot() -> dict[str, float | None]:
            return self._long_value_fit_metrics(observation_tensor, target_tensor)

        snapshots: dict[str, dict[str, float | None]] = {"0": snapshot()}
        gradient_before: list[float] = []
        gradient_after: list[float] = []
        for step in range(1, int(steps) + 1):
            self.long_value_optimizer.zero_grad(set_to_none=True)
            prediction = self.policy.long_value(observation_tensor)
            loss = torch.mean((prediction - target_tensor.detach()) ** 2)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Long-Value Critic loss is non-finite.")
            loss.backward()
            if any(parameter.grad is not None for parameter in protected_parameters):
                raise RuntimeError(
                    "Long-Value Critic update produced Actor/N2/Micro gradients."
                )
            grad_before = float(
                torch.nn.utils.clip_grad_norm_(long_parameters, max_grad_norm)
                .detach()
                .cpu()
            )
            grad_after = _gradient_norm(long_parameters)
            if not math.isfinite(grad_before) or not math.isfinite(grad_after):
                raise FloatingPointError("Long-Value Critic gradient is non-finite.")
            gradient_before.append(grad_before)
            gradient_after.append(grad_after)
            self.long_value_optimizer.step()
            self.training_counters["long_value_optimizer_steps"] += 1
            if not torch.equal(target_tensor, frozen_targets):
                raise RuntimeError(
                    "Long-Value Critic realized-return target changed during updates."
                )
            if step in {10, 25, 50}:
                snapshots[str(step)] = snapshot()

        if any(
            not torch.equal(before, parameter.detach())
            for before, parameter in zip(
                protected_before, protected_parameters, strict=True
            )
        ):
            raise RuntimeError("Long-Value Critic update changed Actor/N2/Micro parameters.")
        target_sha256_after = hashlib.sha256(
            target_tensor.detach().cpu().numpy().tobytes()
        ).hexdigest()
        target_detached = not bool(target_tensor.requires_grad)
        target_frozen = bool(
            torch.equal(target_tensor, frozen_targets)
            and target_sha256_before == target_sha256_after
        )
        if not target_detached or not target_frozen:
            raise RuntimeError("Long-Value Critic target must remain detached and frozen.")
        self.training_counters["long_value_update_count"] += 1
        return {
            "optimizer_steps": int(steps),
            "target_detached": target_detached,
            "target_frozen": target_frozen,
            "target_sha256_before": target_sha256_before,
            "target_sha256_after": target_sha256_after,
            "snapshots": snapshots,
            "grad_norm_before_clip_mean": float(np.mean(gradient_before)),
            "grad_norm_after_clip_mean": float(np.mean(gradient_after)),
        }

    def _long_value_fit_metrics(
        self, observations: Any, value_targets: Any
    ) -> dict[str, float | None]:
        with torch.no_grad():
            predictions = (
                self.policy.long_value(observations)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        targets = value_targets.detach().cpu().numpy().astype(np.float64)
        target_std = float(np.std(targets))
        prediction_std = float(np.std(predictions))
        rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
        return {
            "explained_variance": _explained_variance(targets, predictions),
            "rmse": rmse,
            "mse": float(rmse**2),
            "prediction_std": prediction_std,
            "target_std": target_std,
            "std_ratio": prediction_std / target_std if target_std > 1.0e-12 else None,
            "correlation": _pearson_correlation(targets, predictions),
        }

    @staticmethod
    def _aggregate_update_metrics(
        prefix: str, records: list[dict[str, float]]
    ) -> dict[str, float]:
        if not records:
            raise ValueError(f"{prefix} PPO 没有完成任何 optimizer step。")
        return {
            f"{prefix}_{name}": float(np.mean([record[name] for record in records]))
            for name in records[0]
        }

    def _tensor_batch(
        self, batch: dict[str, np.ndarray], indices: np.ndarray
    ) -> dict[str, Any]:
        return {
            key: torch.as_tensor(value[indices], device=self.device)
            for key, value in batch.items()
            if key
            not in {
                "durations",
                "rewards",
                "old_values",
                "old_next_values",
                "next_observations",
                "requested_durations",
                "dones",
            }
        }

    def checkpoint_payload(
        self,
        *,
        config_raw_sha256: str,
        config_semantic_sha256: str,
        episode_pool: str,
        seed: int,
        best_episode_return: float | None = None,
    ) -> dict[str, Any]:
        """Construct a resumable V15-C2 checkpoint with Trigger protocol metadata."""

        last1 = self.last1_macro_transition_buffer
        return {
            "checkpoint_version": self.checkpoint_version,
            "algorithm_id": self.algorithm_id,
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "macro_critic_lambda_return_lambda": self.macro_critic_lambda_return_lambda,
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "method": self.checkpoint_method,
            "policy_schema": self.policy.architecture_id,
            "architecture_version": int(self.policy.architecture_version),
            "config_contract": self.config_contract,
            "stage1_trigger_only": bool(self.stage1_trigger_only),
            "unified_cache_action": bool(self.unified_cache_action),
            "cardinality_corrected_cache_action": bool(
                self.cardinality_corrected_cache_action
            ),
            "initialization_metadata": dict(self.initialization_metadata),
            "trigger_observation_normalization": (
                self.policy.trigger_observation_normalization_payload()
            ),
            "config_raw_sha256": str(config_raw_sha256),
            "config_semantic_sha256": str(config_semantic_sha256),
            "episode_pool": str(episode_pool),
            "data_protocol": self.context.data_protocol,
            "episode_slots": int(self.context.episode_slots),
            "origin_start": int(self.context.train_origin_start),
            "origin_end": int(self.context.train_origin_end),
            "terminal_origin": int(self.context.terminal_origin),
            "micro_state_dict": self.policy.micro_state_dict(),
            "macro_state_dict": self.policy.macro_state_dict(),
            "macro_cache_trigger_actor_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.policy.macro_cache_trigger_actor.state_dict().items()
            },
            "n2_critic_state_dict": self.policy.n2_critic_state_dict(),
            "macro_critic_state_dict": self.policy.n2_critic_state_dict(),
            "long_value_critic_state_dict": self.policy.long_value_critic_state_dict(),
            "micro_optimizer_state_dict": self.micro_optimizer.state_dict(),
            "macro_optimizer_state_dict": self.macro_optimizer.state_dict(),
            "macro_cache_trigger_actor_optimizer_state_dict": (
                self.macro_cache_trigger_actor_optimizer.state_dict()
            ),
            "macro_critic_optimizer_state_dict": self.macro_critic_optimizer.state_dict(),
            "long_value_optimizer_state_dict": self.long_value_optimizer.state_dict(),
            "value_normalization_states": {"micro": None, "macro": None},
            "training_counters": dict(self.training_counters),
            "last1_macro_transition_buffer": (
                None
                if last1 is None
                else {
                    "episode_index": int(last1["episode_index"]),
                    "observations": np.asarray(last1["observations"], dtype=np.float32).copy(),
                    "next_observations": np.asarray(
                        last1["next_observations"], dtype=np.float32
                    ).copy(),
                    "macro_rewards": np.asarray(
                        last1["macro_rewards"], dtype=np.float32
                    ).copy(),
                    "actual_durations": np.asarray(
                        last1["actual_durations"], dtype=np.int64
                    ).copy(),
                    "dones": np.asarray(last1["dones"], dtype=bool).copy(),
                }
            ),
            "seed": int(seed),
            "best_episode_return": best_episode_return,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        config_raw_sha256: str,
        config_semantic_sha256: str,
        episode_pool: str,
        seed: int,
        best_episode_return: float | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.checkpoint_payload(
                config_raw_sha256=config_raw_sha256,
                config_semantic_sha256=config_semantic_sha256,
                episode_pool=episode_pool,
                seed=seed,
                best_episode_return=best_episode_return,
            ),
            destination,
        )
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        config_raw_sha256: str | None = None,
        config_semantic_sha256: str | None = None,
        episode_pool: str | None = None,
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        """Restore only a structurally complete V16-A version-47 checkpoint."""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping.")
        if (
            payload.get("checkpoint_version") != self.checkpoint_version
            or payload.get("architecture_version")
            != int(self.policy.architecture_version)
        ):
            raise ValueError(
                "Checkpoint architecture/action semantics do not match the active trainer."
            )
        if payload.get("macro_critic_target_mode") != self.macro_critic_target_mode:
            raise ValueError("checkpoint macro_critic_target_mode mismatch.")
        if payload.get("macro_critic_lambda_return_lambda") != (
            self.macro_critic_lambda_return_lambda
        ):
            raise ValueError("checkpoint macro_critic_lambda_return_lambda mismatch.")
        if payload.get("long_value_enabled") is not self.long_value_enabled:
            raise ValueError("checkpoint long_value_enabled mismatch.")
        if payload.get("long_value_target") != self.long_value_target:
            raise ValueError("checkpoint long_value_target mismatch.")
        required = {
            "checkpoint_version": self.checkpoint_version,
            "algorithm_id": self.algorithm_id,
            "method": self.checkpoint_method,
            "policy_schema": self.policy.architecture_id,
            "architecture_version": int(self.policy.architecture_version),
            "unified_cache_action": self.unified_cache_action,
        }
        if self.cardinality_corrected_cache_action:
            required["cardinality_corrected_cache_action"] = True
        for field, expected in required.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"checkpoint {field} mismatch: {payload.get(field)!r} != {expected!r}"
                )
        for field, expected in (
            ("config_raw_sha256", config_raw_sha256),
            ("config_semantic_sha256", config_semantic_sha256),
            ("episode_pool", episode_pool),
        ):
            if expected is not None and payload.get(field) != expected:
                raise ValueError(f"checkpoint {field} mismatch.")
        if payload.get("config_contract") != self.config_contract:
            raise ValueError("checkpoint policy tensor/configuration contract mismatch.")
        if bool(payload.get("stage1_trigger_only", False)) != self.stage1_trigger_only:
            raise ValueError("checkpoint Stage-1 Trigger-only mode mismatch.")
        if bool(payload.get("unified_cache_action", False)) != self.unified_cache_action:
            raise ValueError("checkpoint unified Cache action mode mismatch.")
        if self.cardinality_corrected_cache_action and payload.get(
            "cardinality_corrected_cache_action"
        ) is not True:
            raise ValueError("checkpoint Cache candidate cardinality semantics mismatch.")
        if payload.get("data_protocol") != self.context.data_protocol:
            raise ValueError("checkpoint data protocol mismatch.")
        for field, expected in (
            ("episode_slots", self.context.episode_slots),
            ("origin_start", self.context.train_origin_start),
            ("origin_end", self.context.train_origin_end),
            ("terminal_origin", self.context.terminal_origin),
        ):
            if int(payload.get(field, -1)) != int(expected):
                raise ValueError(f"checkpoint {field} mismatch.")
        macro_state = payload.get("macro_state_dict")
        explicit_n2_state = payload.get("n2_critic_state_dict")
        explicit_critic_state = payload.get("macro_critic_state_dict")
        if not isinstance(macro_state, dict) or not isinstance(explicit_n2_state, dict):
            raise ValueError("checkpoint is missing the scalar N2 Critic state.")
        if not isinstance(explicit_critic_state, dict):
            raise ValueError("checkpoint is missing the scalar Main Critic state.")
        trigger_state = payload.get("macro_cache_trigger_actor_state_dict")
        if not isinstance(trigger_state, dict):
            raise ValueError("checkpoint is missing the independent Trigger Actor state.")
        embedded_n2_state = {
            key.removeprefix("macro_critic."): value
            for key, value in macro_state.items()
            if key.startswith("macro_critic.")
        }
        for explicit_state in (explicit_n2_state, explicit_critic_state):
            if set(embedded_n2_state) != set(explicit_state) or any(
                not torch.equal(embedded_n2_state[key], explicit_state[key])
                for key in embedded_n2_state
            ):
                raise ValueError("checkpoint scalar N2 Critic state copies disagree.")
        self.policy.load_separated_state_dicts(
            macro_state,
            payload["micro_state_dict"],
            payload["long_value_critic_state_dict"],
            trigger_state,
        )
        normalizer = payload.get("trigger_observation_normalization")
        if not isinstance(normalizer, dict) or not isinstance(normalizer.get("active"), bool):
            raise ValueError("checkpoint is missing Trigger normalization metadata.")
        if normalizer["active"]:
            self.policy.set_trigger_observation_normalization(
                normalizer.get("raw_mean", normalizer.get("mean")),
                normalizer.get("raw_std", normalizer.get("std")),
                metadata=normalizer.get("metadata", {}),
            )
        elif self.stage1_trigger_only:
            raise ValueError("V15-C Stage-1 checkpoint lacks active Trigger normalization.")
        self.initialization_metadata = dict(payload.get("initialization_metadata", {}))
        self.macro_optimizer.load_state_dict(payload["macro_optimizer_state_dict"])
        self.macro_cache_trigger_actor_optimizer.load_state_dict(
            payload["macro_cache_trigger_actor_optimizer_state_dict"]
        )
        self.macro_critic_optimizer.load_state_dict(
            payload["macro_critic_optimizer_state_dict"]
        )
        self.micro_optimizer.load_state_dict(payload["micro_optimizer_state_dict"])
        self.long_value_optimizer.load_state_dict(
            payload["long_value_optimizer_state_dict"]
        )
        counters = payload.get("training_counters")
        if not isinstance(counters, dict) or set(counters) != set(self.training_counters):
            raise ValueError("checkpoint training counter schema mismatch.")
        self.training_counters = {
            key: int(value) for key, value in counters.items()
        }
        raw_last1 = payload.get("last1_macro_transition_buffer")
        completed_episodes = int(self.training_counters["completed_episodes"])
        if raw_last1 is None:
            if completed_episodes != 0 and not self.stage1_trigger_only:
                raise ValueError("V14 checkpoint is missing its last1 Macro transition buffer.")
            self.last1_macro_transition_buffer = None
        else:
            expected_keys = {
                "episode_index",
                "observations",
                "next_observations",
                "macro_rewards",
                "actual_durations",
                "dones",
            }
            if not isinstance(raw_last1, dict) or set(raw_last1) != expected_keys:
                raise ValueError("V14 last1 Macro transition buffer schema mismatch.")
            observations = np.asarray(raw_last1["observations"], dtype=np.float32)
            next_observations = np.asarray(
                raw_last1["next_observations"], dtype=np.float32
            )
            rewards = np.asarray(raw_last1["macro_rewards"], dtype=np.float32).reshape(-1)
            durations = np.asarray(raw_last1["actual_durations"], dtype=np.int64).reshape(-1)
            dones = np.asarray(raw_last1["dones"], dtype=bool).reshape(-1)
            size = len(rewards)
            if (
                observations.shape != (size, self.policy.observation_dim)
                or next_observations.shape != observations.shape
                or durations.shape != (size,)
                or dones.shape != (size,)
                or size == 0
                or np.any(durations <= 0)
                or not dones[-1]
                or np.any(dones[:-1])
                or int(raw_last1["episode_index"]) != completed_episodes
                or not all(
                    np.isfinite(item).all()
                    for item in (observations, next_observations, rewards)
                )
            ):
                raise ValueError("V14 last1 Macro transition buffer is invalid.")
            self.last1_macro_transition_buffer = {
                "episode_index": completed_episodes,
                "observations": observations.copy(),
                "next_observations": next_observations.copy(),
                "macro_rewards": rewards.copy(),
                "actual_durations": durations.copy(),
                "dones": dones.copy(),
            }
        if restore_rng:
            if "python_rng_state" not in payload or "numpy_rng_state" not in payload:
                raise ValueError("checkpoint is missing recoverable RNG state.")
            random.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"].cpu())
            cuda_states = payload.get("torch_cuda_rng_states")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        return payload


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _contains_nonfinite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return False
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            return not bool(np.isfinite(value).all())
        if value.dtype.kind == "O":
            return any(_contains_nonfinite(item) for item in value.flat)
        return False
    if isinstance(value, (int, float, np.number)):
        return not math.isfinite(float(value))
    return False


def run_training(
    config_path: str | Path,
    target_episodes: int,
    run_dir: str | Path,
    *,
    episode_pool: str | Path,
    episode_seed_start: int | None = None,
    generate_missing_episodes: bool = False,
    episode_generation_device: str = "cpu",
    actor_learning_rate_override: float | None = None,
    macro_actor_learning_rate_override: float | None = None,
    critic_learning_rate_override: float | None = None,
    seed: int = 0,
    resume: bool = False,
    verbose: bool = False,
    save_macro_transition_trace: bool = False,
    early_gate_episodes: int | None = None,
    initial_checkpoint_path: str | Path | None = None,
    expected_initial_checkpoint_sha256: str | None = None,
    stage1_trigger_only: bool = False,
    trigger_observation_normalization: dict[str, Any] | None = None,
    episode_stop_callback: Any | None = None,
    unified_cache_action: bool = False,
    cardinality_corrected_cache_action: bool = False,
) -> dict[str, Any]:
    """按完整 episode 收集 on-policy 数据，然后分别更新两套策略。"""

    _require_torch()
    if int(target_episodes) <= 0:
        raise ValueError("--epochs 必须为正整数。")
    if early_gate_episodes is not None and (
        int(early_gate_episodes) <= 0
        or int(early_gate_episodes) > int(target_episodes)
        or resume
    ):
        raise ValueError("early gate requires a fresh run and a gate within target_episodes.")
    if unified_cache_action and (stage1_trigger_only or trigger_observation_normalization is not None):
        raise ValueError("Unified Cache Action cannot use Trigger-only mode or normalization.")
    if unified_cache_action and early_gate_episodes is not None:
        raise ValueError("Unified Cache Action uses the 30-episode report gate, not the legacy Trigger gate.")
    if cardinality_corrected_cache_action and not unified_cache_action:
        raise ValueError("Cache cardinality correction requires Unified Cache Action.")
    if save_macro_transition_trace and (resume or episode_seed_start is None):
        raise ValueError("transition-trace reporting requires a fresh run with --episode-seed-start.")
    if initial_checkpoint_path is not None and resume:
        raise ValueError("A downstream warm-start checkpoint is only valid for a fresh run.")
    if stage1_trigger_only and initial_checkpoint_path is None and not resume:
        raise ValueError("V15-C Stage-1 requires an explicit downstream warm-start checkpoint.")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    cfg = project_config.load_config(str(config_file))
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    episode_paths = _resolve_training_episodes(
        project_root=project_root,
        config_path=config_file,
        episode_pool=episode_pool,
        episode_seed_start=episode_seed_start,
        count=int(target_episodes),
        generate_missing=bool(generate_missing_episodes),
        generation_device=episode_generation_device,
    )
    pool_path = _resolve_path(project_root, episode_pool)
    context = load_environment_context(str(config_file), realization_dir=episode_paths[0])
    raw_hash = project_config.raw_config_sha256(config_file)
    semantic_hash = project_config.semantic_config_hash(context.cfg)
    run_path = _resolve_path(project_root, run_dir)
    manifest_path = run_path / RUN_MANIFEST_FILENAME
    checkpoint_path = checkpoint_path_for_resume(run_path, CHECKPOINT_FILENAME)
    _seed_everything(seed)
    trainer = PPOTrainer(
        context,
        actor_learning_rate_override=actor_learning_rate_override,
        macro_actor_learning_rate_override=macro_actor_learning_rate_override,
        critic_learning_rate_override=critic_learning_rate_override,
        capture_macro_transition_trace=save_macro_transition_trace,
        stage1_trigger_only=stage1_trigger_only,
        trigger_observation_normalization=trigger_observation_normalization,
        unified_cache_action=unified_cache_action,
        cardinality_corrected_cache_action=cardinality_corrected_cache_action,
    )
    if initial_checkpoint_path is not None:
        warmstart_loader = (
            trainer.load_v14b_unified_checkpoint
            if unified_cache_action
            else trainer.load_v14b_downstream_checkpoint
        )
        warmstart_loader(
            initial_checkpoint_path,
            expected_sha256=(V14B_LATEST_CHECKPOINT_SHA256
                if expected_initial_checkpoint_sha256 is None
                else expected_initial_checkpoint_sha256),
        )
    completed = 0
    best_return: float | None = None
    if resume:
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("resume 需要现有 run manifest 与 latest checkpoint。")
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        for field, expected in (
            ("config_raw_sha256", raw_hash),
            ("config_semantic_sha256", semantic_hash),
            ("episode_pool", str(pool_path)),
            ("policy_schema", trainer.policy.architecture_id),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"run manifest {field} mismatch。")
        payload = trainer.load_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            restore_rng=True,
        )
        completed = int(trainer.training_counters["completed_episodes"])
        best_return = payload.get("best_episode_return")
        if completed > int(target_episodes):
            raise ValueError("checkpoint completed episodes exceed --epochs。")
    else:
        ensure_fresh_run_dir_is_clean(run_path)
        run_path.mkdir(parents=True, exist_ok=True)
    run_config_path = run_path / "run_config.toml"
    if run_config_path.is_file():
        if project_config.raw_config_sha256(run_config_path) != raw_hash:
            raise ValueError("run_config.toml does not match the canonical config.")
    else:
        shutil.copyfile(config_file, run_config_path)
    ensure_run_artifact_directories(run_path)
    write_dual_timescale_run_manifest(
        manifest_path,
        config_path=config_file,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        seed=seed,
        trainer=trainer,
        context=context,
        target_episodes=target_episodes,
        episode_pool=str(pool_path),
        episode_seed_start=episode_seed_start,
        resume=resume,
    )

    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    critic_diagnostics_path = run_path / "critic_diagnostics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    trace_episode_batches: list[dict[str, Any]] = []
    trace_summary = None
    run_summary_path = run_path / RUN_SUMMARY_FILENAME
    early_stop_reason: str | None = None
    early_stop_detail: str | None = None
    trigger_kl_high_streak = 0
    trigger_p_update_low_streak = 0
    trigger_p_update_high_streak = 0
    first_gate_passed = False
    last_episode_record: dict[str, Any] | None = None
    last_update_record: dict[str, Any] | None = None

    def save_run_summary(early_gate_passed: bool) -> None:
        summary = {
            "experiment_type": "macro_cache_smdp_micro_uav_ppo",
            "algorithm_id": trainer.algorithm_id,
            "architecture_version": int(trainer.policy.architecture_version),
            "checkpoint_version": trainer.checkpoint_version,
            "completed_episodes": int(completed),
            "target_episodes": int(target_episodes),
            "best_episode_return": best_return,
            "early_gate_episodes": early_gate_episodes,
            "early_gate_status": (
                "PASS"
                if early_gate_episodes is not None and early_gate_passed
                else "PENDING"
                if early_gate_episodes is not None and completed < int(early_gate_episodes)
                else "NOT_CONFIGURED"
                if early_gate_episodes is None
                else "FAILED"
            ),
            "early_stop_reason": early_stop_reason,
            "early_stop_detail": early_stop_detail,
            "training_counters": dict(trainer.training_counters),
            "initialization_metadata": dict(trainer.initialization_metadata),
            "stage1_trigger_only": bool(trainer.stage1_trigger_only),
            "unified_cache_action": bool(trainer.unified_cache_action),
            "cardinality_corrected_cache_action": bool(
                trainer.cardinality_corrected_cache_action
            ),
            "trigger_observation_normalization": (
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in trainer.policy
                    .trigger_observation_normalization_payload()
                    .items()
                }
            ),
            "last_episode": last_episode_record,
            "last_update": last_update_record,
        }
        with run_summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")

    for episode_index in range(completed, int(target_episodes)):
        epoch_started_at = time.perf_counter()
        realization_path = episode_paths[episode_index]
        next_context = load_environment_context(
            str(config_file), realization_dir=realization_path
        )
        _assert_realization_context_compatible(context, next_context)
        trainer.context = context = next_context
        try:
            collection = collect_dual_timescale_episode(
                trainer.policy,
                context,
                aligned_train_steps(context),
                trainer.duration_candidates,
                deterministic=False,
                critic_independent_trigger_teacher=not unified_cache_action,
                base_seed=int(seed),
                episode_seed=(
                    int(episode_seed_start if episode_seed_start is not None else seed)
                    + episode_index
                ),
                rollout_index=episode_index + 1,
            )
        except CounterfactualRestoreError as exc:
            early_stop_reason = "COUNTERFACTUAL_STATE_RNG_RESTORE_FAILURE"
            early_stop_detail = str(exc)
            save_run_summary(early_gate_passed=first_gate_passed)
            break
        except FloatingPointError as exc:
            early_stop_reason = "NAN_INF"
            early_stop_detail = str(exc)
            save_run_summary(early_gate_passed=first_gate_passed)
            break
        if _contains_nonfinite(collection.summary):
            early_stop_reason = "NAN_INF"
            early_stop_detail = "Rollout summary contains NaN/Inf."
            save_run_summary(early_gate_passed=first_gate_passed)
            break
        try:
            update_metrics = trainer.update_from_episode(collection)
        except FloatingPointError as exc:
            early_stop_reason = "NAN_INF"
            early_stop_detail = str(exc)
            save_run_summary(early_gate_passed=first_gate_passed)
            break
        if _contains_nonfinite(update_metrics):
            early_stop_reason = "NAN_INF"
            early_stop_detail = "PPO update metrics contain NaN/Inf."
            save_run_summary(early_gate_passed=first_gate_passed)
            break
        training_epoch_runtime = time.perf_counter() - epoch_started_at
        completed = episode_index + 1
        if save_macro_transition_trace:
            trace = trainer.last_macro_transition_trace
            if trace is None:
                raise RuntimeError("Macro transition trace capture was enabled but no trace was produced.")
            trace_count = int(len(trace["macro_reward"]))
            trace_episode = {
                "episode_id": completed,
                "episode_seed": int(episode_seed_start) + episode_index,
                "macro_frame_index": np.arange(trace_count, dtype=np.int64),
                **trace,
            }
            expected_count = int(collection.summary["macro_frame_count"])
            if trace_count != expected_count:
                raise RuntimeError(
                    f"Macro trace E{completed} has {trace_count} transitions; "
                    f"episode summary reports {expected_count}."
                )
            _, episode_trace_check = build_macro_transition_trace_arrays(
                [trace_episode],
                gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
                target_mode=trainer.macro_critic_target_mode,
                lambda_return_lambda=trainer.macro_critic_lambda_return_lambda,
                expected_macro_frame_counts={completed: expected_count},
            )
            if episode_trace_check["transition_count"] <= 0:
                raise RuntimeError(f"Macro trace E{completed} contains no transitions.")
            trace_episode_batches.append(trace_episode)
        episode_return = float(collection.summary["episode_return"])
        if best_return is None or episode_return > float(best_return):
            best_return = episode_return
            trainer.save_checkpoint(
                checkpoint_path_for_write(run_path, BEST_TRAIN_RETURN_CHECKPOINT_FILENAME),
                config_raw_sha256=raw_hash,
                config_semantic_sha256=semantic_hash,
                episode_pool=str(pool_path),
                seed=seed,
                best_episode_return=best_return,
            )
        episode_record = {
            "record_type": "episode",
            "episode": completed,
            "episode_seed": (
                int(episode_seed_start) + episode_index
                if episode_seed_start is not None
                else realization_path.name
            ),
            "training_epoch_runtime_seconds": float(training_epoch_runtime),
            **collection.summary,
        }
        episode_record.update(
            {
                key: value
                for key, value in update_metrics.items()
                if key.startswith(
                    (
                        "macro_actor_advantage",
                        "macro_cache_advantage",
                        "macro_cache_trigger_advantage",
                        "macro_candidate_advantage",
                        "macro_duration_advantage",
                        "macro_trigger_",
                        "trigger_",
                        "macro_candidate_",
                        "macro_critic_",
                        "macro_critic_target",
                        "n2_",
                        "long_value_",
                        "old_value_",
                        "old_next_value_",
                        "td_error_",
                        "pre_update_target_",
                        "post_update_target_",
                        "pre_update_return_",
                        "post_update_return_",
                        "critic_target_",
                    )
                )
                or key in {
                    "macro_actor_advantage_mode",
                    "macro_critic_target_mode",
                    "macro_critic_lambda_return_lambda",
                    "macro_critic_fit_snapshots",
                    "long_value_fit_snapshots",
                    "long_value_target",
                    "long_value_target_detached",
                    "long_value_target_frozen_during_update",
                    "long_value_target_sha256_before",
                    "long_value_target_sha256_after",
                    "macro_approx_kl",
                    "macro_clip_fraction",
                    "joint_logprob_decomposition_max_error",
                    "trigger_advantage_raw_mean",
                    "trigger_advantage_raw_std",
                    "counterfactual_branch_runtime_seconds",
                    "counterfactual_branch_count",
                    "counterfactual_state_restore_valid",
                    "rollout_rng_unchanged",
                    "macro_early_stop",
                    "macro_entropy",
                    "micro_explained_variance",
                    "micro_approx_kl",
                    "micro_clip_fraction",
                    "micro_entropy",
                    "micro_value_rmse",
                    "micro_early_stop",
                }
            }
        )
        optimizer_steps_this_episode = {
            key: int(update_metrics[key])
            for key in (
                "macro_actor_optimizer_steps_this_update",
                "macro_cache_trigger_optimizer_steps_this_update",
                "macro_critic_optimizer_steps_this_update",
                "long_value_optimizer_steps_this_update",
                "micro_actor_optimizer_steps_this_update",
                "micro_critic_optimizer_steps_this_update",
                "formal_non_trigger_optimizer_steps_this_update",
            )
        }
        episode_record.update(optimizer_steps_this_episode)
        update_record = {
            "record_type": "update",
            "episode": completed,
            **update_metrics,
            **optimizer_steps_this_episode,
            **trainer.training_counters,
        }
        if early_gate_episodes is not None:
            trigger_kl_value = float(update_metrics["trigger_kl_mean"])
            trigger_p_update_value = float(update_metrics["trigger_p_update_after"])
            active_fraction_value = float(update_metrics["trigger_credit_active_fraction"])
            trigger_kl_high_streak = (
                trigger_kl_high_streak + 1 if trigger_kl_value > 0.05 else 0
            )
            trigger_p_update_low_streak = (
                trigger_p_update_low_streak + 1
                if trigger_p_update_value < 0.02
                else 0
            )
            trigger_p_update_high_streak = (
                trigger_p_update_high_streak + 1
                if trigger_p_update_value > 0.98
                else 0
            )
            if active_fraction_value < 0.10:
                early_stop_reason = "TRIGGER_ACTIVE_FRACTION_BELOW_0_10"
            elif trigger_kl_high_streak >= TRIGGER_KL_PERSISTENCE_EPISODES:
                early_stop_reason = "TRIGGER_KL_ABOVE_0_05_PERSISTENT"
            elif trigger_p_update_low_streak >= TRIGGER_COLLAPSE_PERSISTENCE_EPISODES:
                early_stop_reason = "P_UPDATE_BELOW_0_02_FOR_3_EPISODES"
            elif trigger_p_update_high_streak >= TRIGGER_COLLAPSE_PERSISTENCE_EPISODES:
                early_stop_reason = "P_UPDATE_ABOVE_0_98_FOR_3_EPISODES"
            if (
                early_stop_reason is None
                and completed >= int(early_gate_episodes)
            ):
                first_gate_passed = True
            gate_status = (
                "FAILED"
                if early_stop_reason is not None and not first_gate_passed
                else "PASS"
                if first_gate_passed
                else "PENDING"
            )
            episode_record["early_gate_status"] = gate_status
            episode_record["early_stop_reason"] = early_stop_reason
            episode_record["trigger_kl_high_streak"] = trigger_kl_high_streak
            episode_record["trigger_p_update_low_streak"] = trigger_p_update_low_streak
            episode_record["trigger_p_update_high_streak"] = trigger_p_update_high_streak
            update_record.update(
                {
                    "early_gate_status": gate_status,
                    "early_stop_reason": early_stop_reason,
                    "trigger_kl_high_streak": trigger_kl_high_streak,
                    "trigger_p_update_low_streak": trigger_p_update_low_streak,
                    "trigger_p_update_high_streak": trigger_p_update_high_streak,
                }
            )
        last_episode_record = episode_record
        last_update_record = update_record
        append_dual_timescale_record(episode_metrics_path, episode_record)
        append_dual_timescale_record(update_metrics_path, update_record)
        critic_diagnostics_record = {
            "record_type": "critic_diagnostics",
            "episode": completed,
            **{
                key: value
                for key, value in update_metrics.items()
                if "critic" in key.lower() or key.startswith("n2_")
            },
        }
        append_dual_timescale_record(
            critic_diagnostics_path, critic_diagnostics_record
        )
        print(
            format_training_epoch_log(
                episode_record,
                target_epochs=int(target_episodes),
                macro_actor_learning_rate=trainer.learning_rates["macro_actor"],
            ),
            flush=True,
        )
        append_dual_timescale_slot_metrics(
            slot_metrics_path, episode=completed, transitions=collection.micro.transitions
        )
        trainer.save_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            seed=seed,
            best_episode_return=best_return,
        )
        write_dual_timescale_run_manifest(
            manifest_path,
            config_path=config_file,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            seed=seed,
            trainer=trainer,
            context=context,
            target_episodes=target_episodes,
            episode_pool=str(pool_path),
            episode_seed_start=episode_seed_start,
            resume=resume,
        )
        save_run_summary(early_gate_passed=first_gate_passed)
        update_training_epoch_return_plot(run_path)
        if save_macro_transition_trace:
            # Persist each validated cumulative trace before starting the next
            # episode so a later report/finalization failure cannot lose raw rows.
            trace_summary = write_macro_transition_trace_artifact(
                run_path,
                trace_episode_batches,
                gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
                target_mode=trainer.macro_critic_target_mode,
                lambda_return_lambda=trainer.macro_critic_lambda_return_lambda,
            )
        if episode_stop_callback is not None:
            callback_stop_reason = episode_stop_callback(
                dict(episode_record), dict(update_record)
            )
            if callback_stop_reason:
                early_stop_reason = str(callback_stop_reason)
                early_stop_detail = "Episode-boundary stop requested by the run-specific callback."
                save_run_summary(early_gate_passed=first_gate_passed)
        if verbose:
            print(json.dumps(update_record, ensure_ascii=False, sort_keys=True))
        if early_stop_reason is not None:
            break
    return {
        "run_dir": str(run_path),
        "completed_episodes": completed,
        "target_episodes": int(target_episodes),
        "checkpoint": str(checkpoint_path),
        "best_episode_return": best_return,
        "macro_transition_trace": trace_summary,
        "early_stop_reason": early_stop_reason,
        "early_gate_passed": first_gate_passed,
    }


def run_synchronized_trigger_training(
    config_path: str | Path,
    target_episodes: int,
    run_dir: str | Path,
    *,
    episode_pool: str | Path,
    episode_seed_start: int,
    realizations_per_update: int,
    macro_actor_learning_rate_override: float,
    seed: int,
    initial_checkpoint_path: str | Path,
    expected_initial_checkpoint_sha256: str,
    trigger_observation_normalization: dict[str, Any],
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run fresh V15-C4 Trigger-only training with synchronized realization batches."""

    _require_torch()
    if int(target_episodes) <= 0 or int(realizations_per_update) != 4:
        raise ValueError("V15-C4 requires a positive episode count and R=4.")
    if int(target_episodes) % int(realizations_per_update) != 0:
        raise ValueError("V15-C4 episode count must be divisible by R=4.")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    cfg = project_config.load_config(str(config_file))
    project_config.validate_for_formal_experiment(cfg)
    project_root = Path(cfg["_meta"]["project_root"])
    episode_paths = _resolve_training_episodes(
        project_root=project_root,
        config_path=config_file,
        episode_pool=episode_pool,
        episode_seed_start=int(episode_seed_start),
        count=int(target_episodes),
        generate_missing=False,
        generation_device="cpu",
    )
    pool_path = _resolve_path(project_root, episode_pool)
    context = load_environment_context(str(config_file), realization_dir=episode_paths[0])
    raw_hash = project_config.raw_config_sha256(config_file)
    semantic_hash = project_config.semantic_config_hash(context.cfg)
    run_path = _resolve_path(project_root, run_dir)
    manifest_path = run_path / RUN_MANIFEST_FILENAME
    checkpoint_path = checkpoint_path_for_resume(run_path, CHECKPOINT_FILENAME)
    _seed_everything(seed)
    trainer = PPOTrainer(
        context,
        macro_actor_learning_rate_override=float(macro_actor_learning_rate_override),
        stage1_trigger_only=True,
        trigger_observation_normalization=trigger_observation_normalization,
    )
    trainer.load_v14b_downstream_checkpoint(
        initial_checkpoint_path,
        expected_sha256=expected_initial_checkpoint_sha256,
    )
    ensure_fresh_run_dir_is_clean(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    run_config_path = run_path / "run_config.toml"
    shutil.copyfile(config_file, run_config_path)
    ensure_run_artifact_directories(run_path)
    write_dual_timescale_run_manifest(
        manifest_path,
        config_path=config_file,
        config_raw_sha256=raw_hash,
        config_semantic_sha256=semantic_hash,
        seed=seed,
        trainer=trainer,
        context=context,
        target_episodes=target_episodes,
        episode_pool=str(pool_path),
        episode_seed_start=episode_seed_start,
        resume=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment_id": "V15-C4_MULTI_REALIZATION_SYNCHRONIZED_TRIGGER",
            "synchronized_realizations_per_update": int(realizations_per_update),
            "synchronized_updates_target": int(target_episodes // realizations_per_update),
            "trigger_updates_during_collection": 0,
            "realization_equal_weighting": True,
            "normalization_source_sha256": trigger_observation_normalization.get(
                "source_sha256"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    episode_metrics_path = run_path / "episode_metrics.jsonl"
    update_metrics_path = run_path / "update_metrics.jsonl"
    critic_diagnostics_path = run_path / "critic_diagnostics.jsonl"
    slot_metrics_path = run_path / SLOT_METRICS_FILENAME
    run_summary_path = run_path / RUN_SUMMARY_FILENAME
    completed = 0
    best_return: float | None = None
    last_episode_record: dict[str, Any] | None = None
    last_update_record: dict[str, Any] | None = None

    def save_run_summary() -> None:
        summary = {
            "experiment_type": "macro_cache_smdp_micro_uav_ppo",
            "algorithm_id": ALGORITHM_ID,
            "architecture_version": int(trainer.policy.architecture_version),
            "checkpoint_version": CHECKPOINT_VERSION,
            "completed_episodes": int(completed),
            "target_episodes": int(target_episodes),
            "synchronized_updates_completed": int(
                trainer.training_counters.get("synchronized_trigger_updates", 0)
            ),
            "realizations_per_update": int(realizations_per_update),
            "best_episode_return": best_return,
            "early_stop_reason": None,
            "training_counters": dict(trainer.training_counters),
            "initialization_metadata": dict(trainer.initialization_metadata),
            "stage1_trigger_only": True,
            "trigger_observation_normalization": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in trainer.policy
                .trigger_observation_normalization_payload()
                .items()
            },
            "last_episode": last_episode_record,
            "last_update": last_update_record,
        }
        run_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    aligned_steps = aligned_train_steps(context)
    durations = trainer.duration_candidates
    synchronized_updates = int(target_episodes // realizations_per_update)
    for update_index in range(synchronized_updates):
        update_started = time.perf_counter()
        first_episode_index = update_index * int(realizations_per_update)
        policy_module = trainer.policy.macro_cache_trigger_actor
        old_policy_version = _module_state_sha256(policy_module)
        old_policy_state = {
            key: value.detach().clone() for key, value in policy_module.state_dict().items()
        }
        collections: list[EpisodeCollection] = []
        episode_contexts: list[Any] = []
        episode_paths_for_group: list[Path] = []
        for realization_offset in range(int(realizations_per_update)):
            episode_index = first_episode_index + realization_offset
            realization_path = episode_paths[episode_index]
            next_context = load_environment_context(
                str(config_file), realization_dir=realization_path
            )
            _assert_realization_context_compatible(context, next_context)
            trainer.context = context = next_context
            try:
                collection = collect_dual_timescale_episode(
                    trainer.policy,
                    context,
                    aligned_train_steps(context),
                    durations,
                    deterministic=False,
                    critic_independent_trigger_teacher=True,
                    base_seed=int(seed),
                    episode_seed=int(episode_seed_start) + episode_index,
                    rollout_index=episode_index + 1,
                )
            except (CounterfactualRestoreError, FloatingPointError):
                raise
            current_state = policy_module.state_dict()
            if _module_state_sha256(policy_module) != old_policy_version or any(
                key not in current_state
                or not torch.equal(old_policy_state[key], current_state[key])
                for key in old_policy_state
            ):
                raise RuntimeError(
                    "Trigger policy changed during a synchronized four-realization collection."
                )
            if _contains_nonfinite(collection.summary):
                raise FloatingPointError("Synchronized rollout summary contains NaN/Inf.")
            if not bool(collection.summary.get("counterfactual_state_restore_valid")):
                raise RuntimeError("COUNTERFACTUAL_STATE_RESTORE_VALID failed in C4 rollout.")
            if not bool(collection.summary.get("rollout_rng_unchanged")):
                raise RuntimeError("Formal rollout RNG changed during C4 counterfactual evaluation.")
            collections.append(collection)
            episode_contexts.append(context)
            episode_paths_for_group.append(realization_path)

        if _module_state_sha256(policy_module) != old_policy_version:
            raise RuntimeError("C4 synchronized collection did not retain one Trigger snapshot.")
        update_metrics = trainer.update_from_synchronized_episodes(collections)
        update_metrics["old_trigger_policy_version"] = old_policy_version
        update_metrics["old_trigger_policy_versions"] = [old_policy_version] * int(
            realizations_per_update
        )
        update_metrics["synchronized_update"] = int(update_index + 1)
        if _contains_nonfinite(update_metrics):
            raise FloatingPointError("Synchronized Trigger PPO update contains NaN/Inf.")

        group_spearman = update_metrics.get("realization_policy_delta_spearman", {})
        group_diagnostics = update_metrics.get("realization_trigger_diagnostics", {})
        for realization_offset, (collection, realization_path) in enumerate(
            zip(collections, episode_paths_for_group, strict=True)
        ):
            episode_index = first_episode_index + realization_offset
            diagnostic = group_diagnostics.get(str(realization_offset), {})
            episode_record = {
                "record_type": "episode",
                "episode": int(episode_index + 1),
                "episode_seed": int(episode_seed_start) + episode_index,
                "synchronized_update": int(update_index + 1),
                "realization_in_update": int(realization_offset + 1),
                "old_trigger_policy_version": old_policy_version,
                "realization_policy_delta_spearman": group_spearman.get(
                    str(realization_offset)
                ),
                **collection.summary,
                "trigger_p_update_after": diagnostic.get("p_update_mean"),
                "trigger_p_update_state_std": diagnostic.get("p_update_state_std"),
                "trigger_entropy": diagnostic.get("entropy"),
                "trigger_credit_active_fraction": diagnostic.get("active_fraction"),
                "trigger_delta_positive_fraction": diagnostic.get(
                    "delta_positive_fraction"
                ),
                "trigger_delta_negative_fraction": diagnostic.get(
                    "delta_negative_fraction"
                ),
                "trigger_policy_delta_spearman": diagnostic.get(
                    "policy_delta_spearman"
                ),
                "trigger_kl_mean": diagnostic.get("kl_mean"),
                "trigger_kl_max": diagnostic.get("kl_max"),
                "trigger_postclip_normalized_obs_abs_max": update_metrics.get(
                    "trigger_postclip_normalized_obs_abs_max"
                ),
                "trigger_preclip_normalized_obs_abs_max": update_metrics.get(
                    "trigger_preclip_normalized_obs_abs_max"
                ),
                "trigger_normalized_obs_clipped_value_fraction": update_metrics.get(
                    "trigger_normalized_obs_clipped_value_fraction"
                ),
            }
            append_dual_timescale_record(episode_metrics_path, episode_record)
            append_dual_timescale_slot_metrics(
                slot_metrics_path,
                episode=int(episode_index + 1),
                transitions=collection.micro.transitions,
            )
            last_episode_record = episode_record

        completed = first_episode_index + int(realizations_per_update)
        group_returns = [float(item.summary["episode_return"]) for item in collections]
        for episode_return in group_returns:
            if best_return is None or episode_return > float(best_return):
                best_return = episode_return
        update_record = {
            "record_type": "update",
            "episode": completed,
            "synchronized_update": int(update_index + 1),
            "episode_seed_start": int(episode_seed_start) + first_episode_index,
            "episode_seed_end": int(episode_seed_start) + completed - 1,
            "realization_seeds": [
                int(episode_seed_start) + first_episode_index + offset
                for offset in range(int(realizations_per_update))
            ],
            "return_mean_of_realizations": float(np.mean(group_returns)),
            "synchronized_collection_runtime_seconds": float(
                time.perf_counter() - update_started
            ),
            **update_metrics,
            **trainer.training_counters,
        }
        append_dual_timescale_record(update_metrics_path, update_record)
        append_dual_timescale_record(
            critic_diagnostics_path,
            {
                "record_type": "critic_diagnostics",
                "episode": completed,
                "synchronized_update": int(update_index + 1),
                "frozen_downstream": True,
            },
        )
        trainer.save_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            seed=seed,
            best_episode_return=best_return,
        )
        if any(
            value == best_return for value in group_returns
        ):
            trainer.save_checkpoint(
                checkpoint_path_for_write(run_path, BEST_TRAIN_RETURN_CHECKPOINT_FILENAME),
                config_raw_sha256=raw_hash,
                config_semantic_sha256=semantic_hash,
                episode_pool=str(pool_path),
                seed=seed,
                best_episode_return=best_return,
            )
        write_dual_timescale_run_manifest(
            manifest_path,
            config_path=config_file,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            seed=seed,
            trainer=trainer,
            context=context,
            target_episodes=target_episodes,
            episode_pool=str(pool_path),
            episode_seed_start=episode_seed_start,
            resume=False,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "experiment_id": "V15-C4_MULTI_REALIZATION_SYNCHRONIZED_TRIGGER",
                "synchronized_realizations_per_update": int(realizations_per_update),
                "synchronized_updates_target": synchronized_updates,
                "trigger_updates_during_collection": 0,
                "realization_equal_weighting": True,
                "normalization_source_sha256": trigger_observation_normalization.get(
                    "source_sha256"
                ),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        last_update_record = update_record
        save_run_summary()
        update_training_epoch_return_plot(run_path)
        if progress_callback is not None:
            progress_callback(int(update_index + 1), int(completed))
        print(
            json.dumps(
                {
                    "sync_update": int(update_index + 1),
                    "episodes_completed": completed,
                    "seeds": update_record["realization_seeds"],
                    "return_mean_of_4": update_record["return_mean_of_realizations"],
                    "spearman": update_metrics.get("trigger_policy_delta_spearman"),
                    "realization_spearman": group_spearman,
                    "accepted_kl_mean": update_metrics.get("trigger_kl_mean"),
                    "accepted_kl_max": update_metrics.get("trigger_kl_max"),
                    "old_trigger_policy_version": old_policy_version,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    return {
        "run_dir": str(run_path),
        "completed_episodes": completed,
        "synchronized_updates_completed": synchronized_updates,
        "target_episodes": int(target_episodes),
        "checkpoint": str(checkpoint_path),
        "best_episode_return": best_return,
        "early_stop_reason": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练双时间尺度 Cache/UAV PPO。")
    parser.add_argument("--config", default=PROPOSED_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-pool", required=True)
    parser.add_argument("--episode-seed-start", type=int, default=None)
    parser.add_argument("--generate-missing-episodes", action="store_true")
    parser.add_argument("--episode-generation-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--macro-actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-macro-transition-trace", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(
        args.config,
        args.epochs,
        args.run_dir,
        episode_pool=args.episode_pool,
        episode_seed_start=args.episode_seed_start,
        generate_missing_episodes=bool(args.generate_missing_episodes),
        episode_generation_device=args.episode_generation_device,
        actor_learning_rate_override=args.actor_learning_rate,
        macro_actor_learning_rate_override=args.macro_actor_learning_rate,
        critic_learning_rate_override=args.critic_learning_rate,
        seed=args.seed,
        resume=bool(args.resume),
        verbose=bool(args.verbose),
        save_macro_transition_trace=bool(args.save_macro_transition_trace),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
