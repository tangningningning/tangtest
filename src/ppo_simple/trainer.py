"""训练 Macro Cache SMDP PPO 与 Micro UAV autoregressive PPO。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
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
    TRAINING_EPOCH_RETURN_PLOT_FILENAME,
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
    EpisodeCollection,
    MacroRolloutBuffer,
    MicroRolloutBuffer,
    collect_dual_timescale_episode,
    compute_gae,
    compute_macro_smdp_gae,
)

CHECKPOINT_METHOD = "PROPOSED"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"
BEST_TRAIN_RETURN_CHECKPOINT_FILENAME = "best_train_return_checkpoint.pt"
CHECKPOINT_VERSION = 53
ALGORITHM_ID = "PROPOSED"
PROPOSED_CONFIG_PATH = "configs/proposed/system_config.toml"
MACRO_ACTOR_ADVANTAGE_MODE = "formal Macro SMDP GAE; joint Cache and sampled termination PPO"
MACRO_CRITIC_TARGET_MODE = "two_step_smdp_td"
MACRO_CRITIC_LAST1_ALPHA = 1.0
FORMAL_TRAINING_EPISODES = 300


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


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _reconcile_run_artifacts_with_checkpoint(
    run_path: str | Path,
    completed_episodes: int,
    *,
    episode_slots: int,
    group_size: int = 4,
    slot_start: int = 0,
) -> dict[str, int]:
    """按提交的 update-group 边界清理至多一组未提交尾记录。"""

    boundary = int(completed_episodes)
    slot_count = int(episode_slots)
    update_group_size = int(group_size)
    first_slot = int(slot_start)
    if boundary < 0 or slot_count <= 0 or update_group_size <= 0:
        raise ValueError("checkpoint boundary, slot count, and group size must be valid")
    if boundary % update_group_size != 0:
        raise ValueError("checkpoint completed_episodes must be on a policy-update group boundary")
    directory = Path(run_path)
    prepared: dict[Path, tuple[list[bytes], int]] = {}

    def read_records(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
        if not path.exists():
            return []
        records: list[tuple[bytes, dict[str, Any]]] = []
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{path.name} line {line_number} is not valid JSONL"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path.name} line {line_number} is not an object")
                records.append((raw_line, record))
        return records

    episode_path = directory / "episode_metrics.jsonl"
    episode_records = read_records(episode_path)
    episode_ids: list[int] = []
    for line_number, (_, record) in enumerate(episode_records, start=1):
        if record.get("record_type") != "episode":
            raise ValueError(f"episode_metrics.jsonl line {line_number} has an invalid record_type")
        episode = record.get("episode")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
            raise ValueError(f"episode_metrics.jsonl line {line_number} has an invalid episode")
        group_id = (episode - 1) // update_group_size + 1
        group_start = (group_id - 1) * update_group_size + 1
        expected_metadata = {
            "policy_update_group_id": group_id,
            "policy_update_group_start_episode": group_start,
            "policy_update_group_end_episode": group_start + update_group_size - 1,
            "policy_update_group_size": update_group_size,
        }
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"episode_metrics.jsonl line {line_number} has invalid group metadata")
        episode_ids.append(episode)
    if any(right <= left for left, right in zip(episode_ids, episode_ids[1:])):
        raise ValueError("episode_metrics.jsonl contains duplicate or out-of-order episodes")
    committed_episodes = [episode for episode in episode_ids if episode <= boundary]
    if committed_episodes != list(range(1, boundary + 1)):
        raise ValueError("episode_metrics.jsonl does not contain every committed episode exactly once")
    episode_tail = [episode for episode in episode_ids if episode > boundary]
    if episode_tail and (
        episode_tail != list(range(boundary + 1, episode_tail[-1] + 1))
        or episode_tail[-1] > boundary + update_group_size
    ):
        raise ValueError("episode_metrics.jsonl has a tail outside one uncommitted group")
    prepared[episode_path] = (
        [raw for raw, record in episode_records if int(record["episode"]) <= boundary],
        len(episode_records) - len(committed_episodes),
    )

    for filename, record_type in (
        ("update_metrics.jsonl", "update"),
        ("critic_diagnostics.jsonl", "critic_diagnostics"),
    ):
        path = directory / filename
        records = read_records(path)
        group_ends: list[int] = []
        for line_number, (_, record) in enumerate(records, start=1):
            if record.get("record_type") != record_type:
                raise ValueError(f"{filename} line {line_number} has an invalid record_type")
            episode = record.get("episode")
            if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
                raise ValueError(f"{filename} line {line_number} has an invalid group end episode")
            group_id = episode // update_group_size
            if (
                episode % update_group_size != 0
                or record.get("policy_update_group_id") != group_id
                or record.get("policy_update_group_start_episode")
                != episode - update_group_size + 1
                or record.get("policy_update_group_end_episode") != episode
                or record.get("policy_update_group_size") != update_group_size
            ):
                raise ValueError(f"{filename} line {line_number} has invalid group metadata")
            group_ends.append(episode)
        if any(right <= left for left, right in zip(group_ends, group_ends[1:])):
            raise ValueError(f"{filename} contains duplicate or out-of-order update groups")
        committed_groups = [episode for episode in group_ends if episode <= boundary]
        expected_groups = list(range(update_group_size, boundary + 1, update_group_size))
        if committed_groups != expected_groups:
            raise ValueError(f"{filename} does not contain every committed update group exactly once")
        group_tail = [episode for episode in group_ends if episode > boundary]
        if group_tail and group_tail != [boundary + update_group_size]:
            raise ValueError(f"{filename} has a tail outside one uncommitted update group")
        keep_lines = [raw for raw, record in records if int(record["episode"]) <= boundary]
        prepared[path] = (keep_lines, len(records) - len(keep_lines))

    slot_path = directory / SLOT_METRICS_FILENAME
    slot_records = read_records(slot_path)
    committed_slot_counts: dict[int, int] = {}
    committed_slot_ids: dict[int, list[int]] = {}
    episode_order: list[int] = []
    previous_key: tuple[int, int] | None = None
    seen_slots: set[tuple[int, int]] = set()
    for line_number, (_, record) in enumerate(slot_records, start=1):
        if record.get("record_type") != "slot":
            raise ValueError(f"{SLOT_METRICS_FILENAME} line {line_number} has an invalid record_type")
        episode = record.get("episode")
        slot = record.get("slot")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode < 1
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < 0
        ):
            raise ValueError(f"{SLOT_METRICS_FILENAME} line {line_number} has an invalid episode/slot")
        key = (episode, slot)
        if previous_key is not None and key <= previous_key:
            raise ValueError(f"{SLOT_METRICS_FILENAME} contains duplicate or out-of-order slots")
        if key in seen_slots:
            raise ValueError(f"{SLOT_METRICS_FILENAME} contains a duplicate episode/slot")
        seen_slots.add(key)
        previous_key = key
        if not episode_order or episode_order[-1] != episode:
            episode_order.append(episode)
        if episode <= boundary:
            committed_slot_counts[episode] = committed_slot_counts.get(episode, 0) + 1
            committed_slot_ids.setdefault(episode, []).append(slot)
    if episode_order:
        committed_episodes = [episode for episode in episode_order if episode <= boundary]
        if committed_episodes != list(range(1, boundary + 1)):
            raise ValueError(f"{SLOT_METRICS_FILENAME} does not contain every committed episode")
        tail_episodes = [episode for episode in episode_order if episode > boundary]
        if tail_episodes and (
            tail_episodes != list(range(boundary + 1, tail_episodes[-1] + 1))
            or tail_episodes[-1] > boundary + update_group_size
        ):
            raise ValueError(f"{SLOT_METRICS_FILENAME} has a tail outside one uncommitted group")
    elif boundary:
        raise ValueError(f"{SLOT_METRICS_FILENAME} is missing committed episode records")
    for episode in range(1, boundary + 1):
        if committed_slot_counts.get(episode, 0) != slot_count:
            raise ValueError(
                f"{SLOT_METRICS_FILENAME} episode {episode} does not contain {slot_count} slots"
            )
        if committed_slot_ids[episode] != list(range(first_slot, first_slot + slot_count)):
            raise ValueError(
                f"{SLOT_METRICS_FILENAME} episode {episode} has a damaged slot sequence"
            )
    if episode_order:
        tail_episode_ids = [episode for episode in episode_order if episode > boundary]
        for episode in tail_episode_ids:
            ids = [slot for (slot_episode, slot) in seen_slots if slot_episode == episode]
            ids.sort()
            if len(ids) > slot_count or ids != list(range(first_slot, first_slot + len(ids))):
                raise ValueError(f"{SLOT_METRICS_FILENAME} has a damaged uncommitted slot tail")
        for episode in tail_episode_ids[:-1]:
            if len([slot for slot_episode, slot in seen_slots if slot_episode == episode]) != slot_count:
                raise ValueError(f"{SLOT_METRICS_FILENAME} has an incomplete non-final tail episode")
    keep_slot_lines = [
        raw for raw, record in slot_records if int(record["episode"]) <= boundary
    ]
    prepared[slot_path] = (keep_slot_lines, len(slot_records) - len(keep_slot_lines))

    removed_counts: dict[str, int] = {}
    for path, (keep_lines, removed_count) in prepared.items():
        removed_counts[path.name] = removed_count
        if removed_count == 0:
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.writelines(keep_lines)
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    return removed_counts


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
        raise RuntimeError("Macro Critic final gradient norm failed current norm matching.")
    if projection_active and abs(cosine_post) > 1.0e-5:
        raise RuntimeError("Macro Critic last1 projection failed orthogonality tolerance.")
    if not all(bool(torch.all(torch.isfinite(item)).item()) for item in combined):
        raise FloatingPointError("Macro Critic combined gradient contains NaN/Inf.")
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
















def _macro_batch(buffer: MacroRolloutBuffer) -> dict[str, np.ndarray]:
    items = buffer.transitions
    if not items:
        raise ValueError("Macro rollout buffer 不能为空。")
    batch_size = len(items)
    max_candidates = max(len(item.cache_candidate_masks) for item in items)
    feature_dim = int(items[0].cache_candidate_features.shape[-1])
    observation_dim = int(items[0].observation.shape[-1])
    max_termination_decisions = max(
        len(item.termination_elapsed_slots) for item in items
    )
    if max_termination_decisions > 4:
        raise ValueError("Adaptive holding rollout exceeds max_slots - 1 sampled decisions.")
    features = np.zeros((batch_size, max_candidates, feature_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_candidates), dtype=bool)
    termination_observations = np.zeros(
        (batch_size, max_termination_decisions, observation_dim), dtype=np.float32
    )
    termination_elapsed_slots = np.zeros(
        (batch_size, max_termination_decisions), dtype=np.int64
    )
    termination_actions = np.zeros(
        (batch_size, max_termination_decisions), dtype=np.int64
    )
    termination_masks = np.zeros(
        (batch_size, max_termination_decisions), dtype=bool
    )
    old_termination_log_probs = np.zeros(
        (batch_size, max_termination_decisions), dtype=np.float32
    )
    old_termination_entropies = np.zeros(
        (batch_size, max_termination_decisions), dtype=np.float32
    )
    for index, item in enumerate(items):
        count = len(item.cache_candidate_masks)
        if item.cache_candidate_features.shape != (count, feature_dim):
            raise ValueError("Macro candidate descriptor shape changed within rollout。")
        features[index, :count] = item.cache_candidate_features
        masks[index, :count] = item.cache_candidate_masks
        termination_count = len(item.termination_elapsed_slots)
        if termination_count:
            if item.termination_observations.shape != (
                termination_count,
                observation_dim,
            ):
                raise ValueError("Termination observation trace shape mismatch.")
            termination_observations[index, :termination_count] = (
                item.termination_observations
            )
            termination_elapsed_slots[index, :termination_count] = (
                item.termination_elapsed_slots
            )
            termination_actions[index, :termination_count] = item.termination_actions
            old_termination_log_probs[index, :termination_count] = (
                item.old_termination_log_probs
            )
            old_termination_entropies[index, :termination_count] = (
                item.old_termination_entropies
            )
            termination_masks[index, :termination_count] = True
    return {
        "observations": np.stack([item.observation for item in items]).astype(np.float32),
        "next_observations": np.stack([item.next_observation for item in items]).astype(np.float32),
        "candidate_features": features,
        "candidate_masks": masks,
        "cache_actions": np.asarray([item.cache_action for item in items], dtype=np.int64),
        "termination_observations": termination_observations,
        "termination_elapsed_slots": termination_elapsed_slots,
        "termination_actions": termination_actions,
        "termination_masks": termination_masks,
        "old_termination_log_probs": old_termination_log_probs,
        "old_termination_entropies": old_termination_entropies,
        "old_cache_log_probs": np.asarray(
            [item.old_cache_log_prob for item in items], dtype=np.float32
        ),
        "old_log_probs": np.asarray([item.old_macro_log_prob for item in items], dtype=np.float32),
        "old_values": np.asarray([item.old_value for item in items], dtype=np.float32),
        "old_next_values": np.asarray(
            [item.old_next_value for item in items], dtype=np.float32
        ),
        "rewards": np.asarray([item.frame_reward for item in items], dtype=np.float32),
        "actual_durations": np.asarray(
            [item.actual_duration for item in items], dtype=np.int64
        ),
        "dones": np.asarray([item.done for item in items], dtype=bool),
        "macro_indices": np.arange(batch_size, dtype=np.int64),
    }






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


def _validate_episode_terminal(dones: np.ndarray, label: str) -> None:
    terminal = np.asarray(dones, dtype=bool).reshape(-1)
    if terminal.size == 0 or not bool(terminal[-1]) or bool(np.any(terminal[:-1])):
        raise ValueError(f"{label} rollout must contain exactly one terminal at its end")


def _concat_micro_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("Micro update requires at least one episode batch")
    max_candidates = max(int(batch["candidate_masks"].shape[-1]) for batch in batches)
    padded: list[dict[str, np.ndarray]] = []
    for batch in batches:
        candidate_count = int(batch["candidate_masks"].shape[-1])
        if candidate_count < max_candidates:
            feature_pad = [(0, 0)] * batch["candidate_features"].ndim
            feature_pad[-2] = (0, max_candidates - candidate_count)
            mask_pad = [(0, 0)] * batch["candidate_masks"].ndim
            mask_pad[-1] = (0, max_candidates - candidate_count)
            batch = {
                **batch,
                "candidate_features": np.pad(
                    batch["candidate_features"], feature_pad, mode="constant"
                ),
                "candidate_masks": np.pad(
                    batch["candidate_masks"], mask_pad, mode="constant"
                ),
            }
        padded.append(batch)
    keys = set(padded[0])
    if any(set(batch) != keys for batch in padded[1:]):
        raise ValueError("Micro episode batches have different fields")
    for key in keys:
        trailing_shape = padded[0][key].shape[1:]
        if any(batch[key].shape[1:] != trailing_shape for batch in padded[1:]):
            raise ValueError(f"Micro episode batch shape mismatch for {key}")
    return {key: np.concatenate([batch[key] for batch in padded], axis=0) for key in keys}


def _concat_macro_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("Macro update requires at least one episode batch")
    max_candidates = max(int(batch["candidate_masks"].shape[-1]) for batch in batches)
    max_termination_decisions = max(
        int(batch["termination_masks"].shape[-1]) for batch in batches
    )
    padded: list[dict[str, np.ndarray]] = []
    for batch in batches:
        batch = dict(batch)
        candidate_count = int(batch["candidate_masks"].shape[-1])
        if candidate_count < max_candidates:
            feature_pad = [(0, 0)] * batch["candidate_features"].ndim
            feature_pad[-2] = (0, max_candidates - candidate_count)
            mask_pad = [(0, 0)] * batch["candidate_masks"].ndim
            mask_pad[-1] = (0, max_candidates - candidate_count)
            batch["candidate_features"] = np.pad(
                batch["candidate_features"], feature_pad, mode="constant"
            )
            batch["candidate_masks"] = np.pad(
                batch["candidate_masks"], mask_pad, mode="constant"
            )
        decision_count = int(batch["termination_masks"].shape[-1])
        if decision_count < max_termination_decisions:
            decision_pad = (
                (0, 0),
                (0, max_termination_decisions - decision_count),
            )
            observation_pad = (
                (0, 0),
                (0, max_termination_decisions - decision_count),
                (0, 0),
            )
            for key in (
                "termination_elapsed_slots",
                "termination_actions",
                "termination_masks",
                "old_termination_log_probs",
                "old_termination_entropies",
            ):
                batch[key] = np.pad(batch[key], decision_pad, mode="constant")
            batch["termination_observations"] = np.pad(
                batch["termination_observations"], observation_pad, mode="constant"
            )
        padded.append(batch)
    keys = set(padded[0])
    if any(set(batch) != keys for batch in padded[1:]):
        raise ValueError("Macro episode batches have different fields")
    for key in keys:
        trailing_shape = padded[0][key].shape[1:]
        if any(batch[key].shape[1:] != trailing_shape for batch in padded[1:]):
            raise ValueError(f"Macro episode batch shape mismatch for {key}")
    combined = {key: np.concatenate([batch[key] for batch in padded], axis=0) for key in keys}
    combined["macro_indices"] = np.arange(len(combined["rewards"]), dtype=np.int64)
    return combined


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
        policy_update_total: int = FORMAL_TRAINING_EPISODES // 4,
        capture_macro_transition_trace: bool = False,
        unified_cache_action: bool = True,
        cardinality_corrected_cache_action: bool = True,
    ) -> None:
        _require_torch()
        self.context = context
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable。")
        self.config = context.cfg["algorithm"]["ppo"]
        self.realizations_per_policy_update = int(
            self.config.get("realizations_per_policy_update", 4)
        )
        if self.realizations_per_policy_update != 4:
            raise ValueError("PROPOSED formal training requires groups of four realizations.")
        self.policy_update_total = int(policy_update_total)
        if self.policy_update_total <= 0:
            raise ValueError("policy_update_total must be positive")
        self.macro_actor_learning_rate_schedule = str(
            self.config.get("macro_actor_learning_rate_schedule", "")
        )
        if self.macro_actor_learning_rate_schedule != "cosine":
            raise ValueError("PROPOSED Macro Actor requires the cosine learning-rate schedule.")
        self.unified_cache_action = bool(unified_cache_action)
        self.cardinality_corrected_cache_action = bool(
            cardinality_corrected_cache_action
        )
        if not self.unified_cache_action or not self.cardinality_corrected_cache_action:
            raise ValueError(
                "The PROPOSED trainer supports only the unified Cache candidate action with cardinality correction."
            )
        self.checkpoint_version = CHECKPOINT_VERSION
        self.algorithm_id = ALGORITHM_ID
        self.checkpoint_method = CHECKPOINT_METHOD
        self.capture_macro_transition_trace = bool(capture_macro_transition_trace)
        self.last_macro_transition_trace: dict[str, np.ndarray] | None = None
        self.last_macro_transition_traces: list[dict[str, np.ndarray]] = []
        self.initialization_metadata: dict[str, Any] = {}
        self.macro_critic_target_mode = str(
            self.config.get("macro_critic_target_mode", MACRO_CRITIC_TARGET_MODE)
        )
        if self.macro_critic_target_mode != MACRO_CRITIC_TARGET_MODE:
            raise ValueError(
                "The PROPOSED mainline requires macro_critic_target_mode = two_step_smdp_td."
            )
        self.long_value_enabled = bool(self.config.get("long_value_enabled", False))
        self.long_value_target = str(
            self.config.get("long_value_target", "realized_return_to_go")
        )
        if not self.long_value_enabled:
            raise ValueError("The PROPOSED mainline requires long_value_enabled = true.")
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
        self.adaptive_holding_enabled = bool(
            self.config.get("adaptive_holding_enabled", False)
        )
        self.adaptive_holding_min_slots = int(
            self.config.get("adaptive_holding_min_slots", 0)
        )
        self.adaptive_holding_max_slots = int(
            self.config.get("adaptive_holding_max_slots", 0)
        )
        if (
            not self.adaptive_holding_enabled
            or self.adaptive_holding_min_slots <= 0
            or self.adaptive_holding_max_slots < self.adaptive_holding_min_slots
            or self.adaptive_holding_max_slots
            > int(context.observation_spec.prediction_horizon)
        ):
            raise ValueError("PROPOSED adaptive holding contract is invalid.")
        tensor_spec = PrimitiveTensorSpec.create(context)
        macro_feature_dim = 2 * int(np.asarray(context.initial_cache).size) + 1
        micro_feature_dim = int(tensor_spec.uav_feature_dim) + 2
        self.hidden_sizes = tuple(int(value) for value in self.config["hidden_sizes"])
        self.policy = DualTimescaleActorCritic(
            int(context.observation_spec.observation_dim),
            macro_feature_dim,
            micro_feature_dim,
            self.adaptive_holding_min_slots,
            self.adaptive_holding_max_slots,
            self.hidden_sizes,
            unified_cache_action=self.unified_cache_action,
            cardinality_corrected_cache_action=self.cardinality_corrected_cache_action,
        ).to(self.device)
        actor_lr = float(
            actor_learning_rate_override
            if actor_learning_rate_override is not None
            else self.config.get("actor_learning_rate", self.config["learning_rate"])
        )
        macro_actor_lr_initial = float(
            self.config.get("macro_actor_learning_rate", actor_lr)
        )
        if macro_actor_learning_rate_override is not None:
            macro_actor_lr_initial = float(macro_actor_learning_rate_override)
        macro_actor_lr_final = float(
            self.config.get("macro_actor_final_learning_rate", 1.0e-4)
        )
        if (
            not math.isfinite(macro_actor_lr_initial)
            or macro_actor_lr_initial <= 0.0
            or not math.isfinite(macro_actor_lr_final)
            or macro_actor_lr_final <= 0.0
            or macro_actor_lr_initial < macro_actor_lr_final
        ):
            raise ValueError("Macro Actor cosine learning-rate endpoints are invalid.")
        macro_actor_lr = macro_actor_lr_initial
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
        if macro_actor_parameter_ids.intersection(macro_critic_parameter_ids):
            raise RuntimeError("Macro Actor and scalar Critic parameters must be disjoint.")
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
            "macro_actor_initial": macro_actor_lr_initial,
            "macro_actor_final": macro_actor_lr_final,
            "micro_actor": actor_lr,
            "critic": critic_lr,
            "macro_critic": macro_critic_lr,
            "micro_critic": micro_critic_lr,
            "long_value": long_value_lr,
        }
        self.training_counters = {
            "completed_episodes": 0,
            "policy_update_count": 0,
            "micro_optimizer_steps": 0,
            "macro_optimizer_steps": 0,
            "micro_actor_optimizer_steps": 0,
            "micro_critic_optimizer_steps": 0,
            "macro_actor_optimizer_steps": 0,
            "macro_critic_optimizer_steps": 0,
            "long_value_optimizer_steps": 0,
            "long_value_update_count": 0,
            "micro_update_count": 0,
            "macro_update_count": 0,
            "environment_executions": 0,
            "matching_calls": 0,
            "kkt_calls": 0,
        }
        self.last1_macro_transition_buffer: dict[str, Any] | None = None



    @property
    def config_contract(self) -> dict[str, Any]:
        return {
            "observation_dim": int(self.policy.observation_dim),
            "macro_cache_feature_dim": int(self.policy.macro_cache_feature_dim),
            "micro_uav_feature_dim": int(self.policy.micro_uav_feature_dim),
            "adaptive_holding_enabled": self.adaptive_holding_enabled,
            "adaptive_holding_min_slots": self.adaptive_holding_min_slots,
            "adaptive_holding_max_slots": self.adaptive_holding_max_slots,
            "prediction_horizon": int(self.context.observation_spec.prediction_horizon),
            "hidden_sizes": list(self.hidden_sizes),
            "policy_schema": str(self.policy.architecture_id),
            "architecture_version": int(self.policy.architecture_version),
            "algorithm_id": self.algorithm_id,
            "macro_actor_advantage_mode": MACRO_ACTOR_ADVANTAGE_MODE,
            "realizations_per_policy_update": self.realizations_per_policy_update,
            "policy_update_total": self.policy_update_total,
            "macro_actor_learning_rate_schedule": self.macro_actor_learning_rate_schedule,
            "macro_actor_learning_rate_initial": self.learning_rates[
                "macro_actor_initial"
            ],
            "macro_actor_learning_rate_final": self.learning_rates[
                "macro_actor_final"
            ],
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "macro_critic_value_loss_coefficient": float(
                self.config["value_loss_coefficient"]
            ),
            "macro_critic_architecture": "400-128-128-1 (SiLU)",
            "macro_critic_optimizer": "single Adam",
            "macro_critic_optimizer_epsilon": 1.0e-5,
            "macro_critic_group_objective": "0.01 times raw current-group and previous-group-last1 MSE",
            "macro_critic_conflict_projection": "last1 norm-matched; current-priority projection; final norm matched to current",
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "long_value_learning_rate": self.learning_rates["long_value"],
            "long_value_steps_per_update": self.long_value_steps_per_update,
            "long_value_max_grad_norm": self.long_value_max_grad_norm,
            "cache_action_semantics": "single masked categorical over KEEP candidate 0 and legal UPDATE candidates; holding termination uses post-slot observation and elapsed fraction",
            "cardinality_corrected_cache_action": True,
            "cache_candidate_probability_semantics": "legal UPDATE logits subtract log of the legal nonzero candidate count",
            "unified_cache_action": True,
            "micro_group_entropy_floor_enabled": True,
            "micro_group_probability_soft_cap": float(
                self.config["micro_group_probability_soft_cap"]
            ),
            "micro_group_entropy_floor": self._micro_group_entropy_floor(),
            "micro_group_entropy_floor_coefficient": float(
                self.config["micro_group_entropy_floor_coefficient"]
            ),
            "micro_target_kl": float(self.config["micro_target_kl"]),
            "macro_target_kl": float(self.config["macro_target_kl"]),
            "micro_actor_objective": (
                "single joint UAV0/UAV1 PPO clipped surrogate plus "
                "STAY/MOVE group entropy-floor regularization"
            ),
        }

    def _micro_group_entropy_floor(self) -> float:
        probability_cap = float(self.config["micro_group_probability_soft_cap"])
        return float(
            -probability_cap * math.log(probability_cap)
            - (1.0 - probability_cap) * math.log(1.0 - probability_cap)
        )

    def _macro_actor_learning_rate_for_update(self, update_index: int) -> float:
        last_index = max(self.policy_update_total - 1, 1)
        progress = min(max(int(update_index), 0), last_index) / last_index
        initial = float(self.learning_rates["macro_actor_initial"])
        final = float(self.learning_rates["macro_actor_final"])
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * progress))

    def _apply_macro_actor_learning_rate(self, update_index: int) -> float:
        rate = self._macro_actor_learning_rate_for_update(update_index)
        for parameter_group in self.macro_optimizer.param_groups:
            parameter_group["lr"] = rate
        self.learning_rates["macro_actor"] = rate
        return rate

    def update_from_episodes(
        self, collections: list[EpisodeCollection]
    ) -> dict[str, Any]:
        """用同一冻结策略采样的四条 rollout 执行一次 PROPOSED 更新。"""

        group_size = self.realizations_per_policy_update
        if len(collections) != group_size:
            raise ValueError(f"policy update requires exactly {group_size} realizations")
        episode_start = int(self.training_counters["completed_episodes"]) + 1
        if (episode_start - 1) % group_size != 0:
            raise RuntimeError("training state is not on a policy-update group boundary")
        for collection in collections:
            _validate_episode_terminal(
                np.asarray([item.done for item in collection.micro.transitions], dtype=bool),
                "Micro",
            )
            _validate_episode_terminal(
                np.asarray([item.done for item in collection.macro.transitions], dtype=bool),
                "Macro",
            )

        applied_learning_rate = self._apply_macro_actor_learning_rate(
            int(self.training_counters["policy_update_count"])
        )
        micro_metrics = self._update_micro([collection.micro for collection in collections])
        macro_metrics = self._update_macro(
            [collection.macro for collection in collections],
            episode_start_index=episode_start,
        )
        self.training_counters["completed_episodes"] += group_size
        self.training_counters["policy_update_count"] += 1
        for collection in collections:
            self.training_counters["environment_executions"] += int(
                collection.summary["environment_execution_count"]
            )
            self.training_counters["matching_calls"] += int(
                collection.summary["matching_call_count"]
            )
            self.training_counters["kkt_calls"] += int(
                collection.summary["kkt_call_count"]
            )
        update_index = int(self.training_counters["policy_update_count"])
        episode_end = episode_start + group_size - 1
        return {
            **micro_metrics,
            **macro_metrics,
            "policy_update_group_id": update_index,
            "policy_update_index": update_index,
            "policy_update_total": self.policy_update_total,
            "policy_update_group_start_episode": episode_start,
            "policy_update_group_end_episode": episode_end,
            "policy_update_group_size": group_size,
            "policy_update_group_episode_indices": list(range(episode_start, episode_end + 1)),
            "policy_update_group_seeds": [
                collection.summary.get("episode_seed") for collection in collections
            ],
            "macro_actor_learning_rate_applied": applied_learning_rate,
            "macro_actor_learning_rate_schedule": self.macro_actor_learning_rate_schedule,
            "combined_macro_advantage_mean": float(macro_metrics["macro_adv_mean"]),
            "combined_macro_advantage_std": float(macro_metrics["macro_adv_std"]),
            "combined_micro_advantage_mean": float(micro_metrics["micro_adv_mean"]),
            "combined_micro_advantage_std": float(micro_metrics["micro_adv_std"]),
            **self.training_counters,
        }




    def _update_micro(self, buffers: list[MicroRolloutBuffer]) -> dict[str, Any]:
        episode_batches = [_micro_batch(buffer) for buffer in buffers]
        episode_advantages: list[np.ndarray] = []
        episode_returns: list[np.ndarray] = []
        for episode_batch in episode_batches:
            _validate_episode_terminal(episode_batch["dones"], "Micro")
            advantages, returns = compute_gae(
                episode_batch["rewards"],
                episode_batch["old_values"].astype(np.float64),
                episode_batch["dones"],
                0.0,
                float(self.config["gamma"]),
                float(self.config["gae_lambda"]),
            )
            episode_advantages.append(advantages)
            episode_returns.append(returns)
        batch = _concat_micro_batches(episode_batches)
        count = len(batch["rewards"])
        advantages = np.concatenate(episode_advantages).astype(np.float32, copy=False)
        returns = np.concatenate(episode_returns).astype(np.float32, copy=False)
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
        target_kl = float(self.config["micro_target_kl"])
        group_entropy_floor = self._micro_group_entropy_floor()
        group_entropy_floor_coefficient = float(
            self.config["micro_group_entropy_floor_coefficient"]
        )
        max_grad_norm = float(self.config["max_grad_norm"])
        records: list[dict[str, float]] = []
        stopped = False
        before_tensors = self._tensor_batch(batch, np.arange(count, dtype=np.int64))
        with torch.no_grad():
            before_evaluation = self.policy.evaluate_micro_joint(
                before_tensors["observations"],
                before_tensors["candidate_features"],
                before_tensors["candidate_masks"],
                before_tensors["actions"],
                before_tensors["conditions"],
            )
            uav0_group_entropy_before = float(
                before_evaluation.uav0_group_entropy.mean().detach().cpu()
            )
            uav1_group_entropy_before = float(
                before_evaluation.uav1_group_entropy.mean().detach().cpu()
            )
            group_probability_max_before = float(
                torch.stack(
                    (
                        before_evaluation.uav0_stay_probability,
                        before_evaluation.uav0_move_probability,
                        before_evaluation.uav1_stay_probability,
                        before_evaluation.uav1_move_probability,
                    ),
                    dim=-1,
                ).max().detach().cpu()
            )
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
                group_entropies = torch.stack(
                    (evaluated.uav0_group_entropy, evaluated.uav1_group_entropy),
                    dim=-1,
                )
                group_entropy_floor_loss = group_entropy_floor_coefficient * torch.relu(
                    group_entropy_floor - group_entropies
                ).mean()
                group_entropy_floor_active_fraction = torch.mean(
                    (group_entropies < group_entropy_floor).to(group_entropies.dtype)
                )
                loss = (
                    actor_loss
                    + value_coefficient * value_loss
                    + group_entropy_floor_loss
                    - entropy_coefficient * entropy
                )
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
                        "group_entropy_floor_loss_mean": float(
                            group_entropy_floor_loss.detach().cpu()
                        ),
                        "group_entropy_floor_active_fraction": float(
                            group_entropy_floor_active_fraction.detach().cpu()
                        ),
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
            after_evaluation = self.policy.evaluate_micro_joint(
                before_tensors["observations"],
                before_tensors["candidate_features"],
                before_tensors["candidate_masks"],
                before_tensors["actions"],
                before_tensors["conditions"],
            )
            uav0_group_entropy_after = float(
                after_evaluation.uav0_group_entropy.mean().detach().cpu()
            )
            uav1_group_entropy_after = float(
                after_evaluation.uav1_group_entropy.mean().detach().cpu()
            )
            group_probability_max_after = float(
                torch.stack(
                    (
                        after_evaluation.uav0_stay_probability,
                        after_evaluation.uav0_move_probability,
                        after_evaluation.uav1_stay_probability,
                        after_evaluation.uav1_move_probability,
                    ),
                    dim=-1,
                ).max().detach().cpu()
            )
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
                "micro_target_kl": float(target_kl),
                "micro_group_entropy_floor": float(group_entropy_floor),
                "micro_group_entropy_floor_coefficient": float(
                    group_entropy_floor_coefficient
                ),
                "micro_uav0_group_entropy_before_update": uav0_group_entropy_before,
                "micro_uav1_group_entropy_before_update": uav1_group_entropy_before,
                "micro_uav0_group_entropy_after_update": uav0_group_entropy_after,
                "micro_uav1_group_entropy_after_update": uav1_group_entropy_after,
                "micro_group_probability_max_before_update": group_probability_max_before,
                "micro_group_probability_max_after_update": group_probability_max_after,
            }
        )
        return result

    def _update_macro(
        self,
        buffers: list[MacroRolloutBuffer],
        *,
        episode_start_index: int,
    ) -> dict[str, Any]:
        episode_batches = [_macro_batch(buffer) for buffer in buffers]
        episode_advantages: list[np.ndarray] = []
        episode_critic_targets: list[np.ndarray] = []
        episode_return_to_go: list[np.ndarray] = []
        for episode_batch in episode_batches:
            _validate_episode_terminal(episode_batch["dones"], "Macro")
            raw_advantage, _ = compute_macro_smdp_gae(
                episode_batch["rewards"],
                episode_batch["old_values"].astype(np.float64),
                episode_batch["old_next_values"],
                episode_batch["actual_durations"],
                episode_batch["dones"],
                float(self.config["gamma"]),
                float(self.config["gae_lambda"]),
            )
            episode_advantages.append(raw_advantage)
            episode_critic_targets.append(
                build_macro_n2_target(
                    episode_batch["rewards"],
                    episode_batch["old_values"],
                    episode_batch["actual_durations"],
                    episode_batch["dones"],
                    float(self.config["gamma"]),
                ).astype(np.float32)
            )
            episode_return_to_go.append(
                _realized_smdp_return_to_go(
                    episode_batch["rewards"],
                    episode_batch["actual_durations"],
                    episode_batch["dones"],
                    float(self.config["gamma"]),
                )
            )
        batch = _concat_macro_batches(episode_batches)
        count = len(batch["rewards"])
        episode_end_index = int(episode_start_index) + len(buffers) - 1
        last1_buffer = self.last1_macro_transition_buffer
        if last1_buffer is not None and int(last1_buffer["episode_index"]) != int(episode_start_index) - 1:
            raise RuntimeError("Last1 buffer must contain exactly the preceding episode.")
        next_values = batch["old_next_values"].astype(np.float32, copy=True)
        macro_actor_advantage = np.concatenate(episode_advantages).astype(
            np.float32, copy=False
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
        macro_critic_target = np.concatenate(episode_critic_targets).astype(
            np.float32, copy=False
        )
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
        realized_return_to_go = np.concatenate(episode_return_to_go).astype(
            np.float32, copy=False
        )
        self.last_macro_transition_trace = None
        self.last_macro_transition_traces = []
        if self.capture_macro_transition_trace:
            for episode_batch, episode_target, episode_rtg in zip(
                episode_batches,
                episode_critic_targets,
                episode_return_to_go,
                strict=True,
            ):
                episode_count = len(episode_batch["rewards"])
                episode_old_values = episode_batch["old_values"].astype(
                    np.float32, copy=True
                )
                bootstrap_old_value = np.zeros(episode_count, dtype=np.float32)
                bootstrap_frame_index = np.full(episode_count, -1, dtype=np.int64)
                for index in range(episode_count):
                    if (
                        episode_batch["dones"][index]
                        or index + 1 >= episode_count
                        or episode_batch["dones"][index + 1]
                    ):
                        continue
                    bootstrap_old_value[index] = episode_old_values[index + 2]
                    bootstrap_frame_index[index] = index + 2
                self.last_macro_transition_traces.append(
                    {
                        "observation": episode_batch["observations"].astype(
                            np.float32, copy=True
                        ),
                        "next_observation": episode_batch["next_observations"].astype(
                            np.float32, copy=True
                        ),
                        "macro_reward": episode_batch["rewards"].astype(
                            np.float32, copy=True
                        ),
                        "actual_duration": episode_batch["actual_durations"].astype(
                            np.int64, copy=True
                        ),
                        "termination_observations": episode_batch[
                            "termination_observations"
                        ].astype(np.float32, copy=True),
                        "termination_elapsed_slots": episode_batch[
                            "termination_elapsed_slots"
                        ].astype(np.int64, copy=True),
                        "termination_actions": episode_batch[
                            "termination_actions"
                        ].astype(np.int64, copy=True),
                        "termination_masks": episode_batch[
                            "termination_masks"
                        ].astype(bool, copy=True),
                        "old_termination_log_probs": episode_batch[
                            "old_termination_log_probs"
                        ].astype(np.float32, copy=True),
                        "old_termination_entropies": episode_batch[
                            "old_termination_entropies"
                        ].astype(np.float32, copy=True),
                        "done": episode_batch["dones"].astype(bool, copy=True),
                        "old_value": episode_old_values,
                        "old_next_value": episode_batch["old_next_values"].astype(
                            np.float32, copy=True
                        ),
                        "critic_training_target": episode_target.copy(),
                        "realized_return_to_go": episode_rtg.copy(),
                        "n2_target": episode_target.copy(),
                        "n2_bootstrap_old_value": bootstrap_old_value,
                        "n2_bootstrap_frame_index": bootstrap_frame_index,
                    }
                )
        batch["macro_critic_targets"] = macro_critic_target.copy()
        selected_groups = batch["cache_actions"] > 0
        clip = float(self.config["clip_epsilon"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        target_kl = float(self.config["macro_target_kl"])
        max_grad_norm = float(self.config["max_grad_norm"])
        critic_steps = int(self.config.get("macro_critic_steps_per_update", 50))
        if critic_steps != 50:
            raise ValueError("Macro Critic update budget is fixed at 50 iterations.")
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
                tensors["termination_observations"],
                tensors["termination_elapsed_slots"],
                tensors["termination_actions"],
                tensors["termination_masks"],
            )
            decomposition_error = float(
                components.joint_logprob_decomposition_error.max().detach().cpu()
            )
            max_decomposition_error = max(max_decomposition_error, decomposition_error)
            if _epoch == 0:
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
            cache_log_ratio = (
                components.cache_log_prob - tensors["old_cache_log_probs"]
            )
            cache_ratio = torch.exp(cache_log_ratio)
            cache_kl = torch.mean((cache_ratio - 1.0) - cache_log_ratio)
            cache_clip_fraction = torch.mean(
                (torch.abs(cache_ratio - 1.0) > clip).float()
            )
            decision_mask = tensors["termination_masks"]
            termination_log_ratio = (
                components.termination_log_probs
                - tensors["old_termination_log_probs"]
            )
            if bool(torch.any(decision_mask).item()):
                selected_termination_log_ratio = termination_log_ratio[decision_mask]
                selected_termination_ratio = torch.exp(
                    selected_termination_log_ratio
                )
                termination_kl = torch.mean(
                    (selected_termination_ratio - 1.0)
                    - selected_termination_log_ratio
                )
                termination_clip_fraction = torch.mean(
                    (torch.abs(selected_termination_ratio - 1.0) > clip).float()
                )
            else:
                termination_kl = torch.zeros((), dtype=cache_kl.dtype, device=self.device)
                termination_clip_fraction = torch.zeros(
                    (), dtype=cache_kl.dtype, device=self.device
                )

            joint_log_ratio = components.joint_log_prob - tensors["old_log_probs"]
            joint_ratio = torch.exp(joint_log_ratio)
            actor_loss = _clipped_ppo_surrogate_loss(
                joint_ratio, tensors["macro_actor_advantage"], clip
            )
            joint_kl = torch.mean((joint_ratio - 1.0) - joint_log_ratio)
            joint_clip_fraction = torch.mean(
                (torch.abs(joint_ratio - 1.0) > clip).float()
            )
            value_loss = torch.mean(
                (components.value - tensors["macro_critic_targets"]) ** 2
            )
            entropy_mean = (
                components.cache_entropy + components.termination_entropy_sum
            ).mean()
            loss = actor_loss - entropy_coefficient * entropy_mean
            self.macro_optimizer.zero_grad(set_to_none=True)
            self.macro_critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            macro_actor_parameters = self.policy.macro_actor_parameters()
            macro_critic_parameters = self.policy.macro_critic_parameters()
            if any(parameter.grad is not None for parameter in macro_critic_parameters):
                raise RuntimeError("Macro actor PPO update produced Macro Critic gradients.")
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
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_mean.detach().cpu()),
                    "joint_approx_kl": float(joint_kl.detach().cpu()),
                    "joint_clip_fraction": float(joint_clip_fraction.detach().cpu()),
                    "cache_ratio_approx_kl": float(cache_kl.detach().cpu()),
                    "cache_ratio_clip_fraction": float(cache_clip_fraction.detach().cpu()),
                    "termination_ratio_approx_kl": float(termination_kl.detach().cpu()),
                    "termination_ratio_clip_fraction": float(
                        termination_clip_fraction.detach().cpu()
                    ),
                    "actor_grad_norm_before_clip": macro_actor_before_clip,
                    "actor_grad_norm_after_clip": macro_actor_after_clip,
                    "critic_grad_norm_before_clip": 0.0,
                    "critic_grad_norm_after_clip": 0.0,
                }
            )
            if target_kl > 0.0 and records[-1]["joint_approx_kl"] > 1.5 * target_kl:
                stopped = True
                break

        diagnostic_tensors = self._tensor_batch(batch, all_indices)
        with torch.no_grad():
            diagnostic_components = self.policy.evaluate_macro_components(
                diagnostic_tensors["observations"],
                diagnostic_tensors["candidate_features"],
                diagnostic_tensors["candidate_masks"],
                diagnostic_tensors["cache_actions"],
                diagnostic_tensors["termination_observations"],
                diagnostic_tensors["termination_elapsed_slots"],
                diagnostic_tensors["termination_actions"],
                diagnostic_tensors["termination_masks"],
            )
            joint_log_ratio_abs = torch.abs(
                diagnostic_components.joint_log_prob
                - diagnostic_tensors["old_log_probs"]
            ).detach().cpu().numpy()
        duration_values = np.asarray(batch["actual_durations"], dtype=np.int64)
        macro_joint_log_ratio_abs_mean_by_duration = {
            f"D={duration}": (
                float(np.mean(joint_log_ratio_abs[duration_values == duration]))
                if bool(np.any(duration_values == duration))
                else None
            )
            for duration in range(1, 6)
        }
        sampled_termination_decisions = np.sum(
            np.asarray(batch["termination_masks"], dtype=bool), axis=1
        ).astype(np.int64)

        self.training_counters["macro_update_count"] += 1
        result = self._aggregate_update_metrics("macro", records)
        last_episode_batch = episode_batches[-1]
        self.last1_macro_transition_buffer = {
            "episode_index": episode_end_index,
            "observations": last_episode_batch["observations"].astype(
                np.float32, copy=True
            ),
            "next_observations": last_episode_batch["next_observations"].astype(
                np.float32, copy=True
            ),
            "macro_rewards": last_episode_batch["rewards"].astype(
                np.float32, copy=True
            ),
            "actual_durations": last_episode_batch["actual_durations"].astype(
                np.int64, copy=True
            ),
            "dones": last_episode_batch["dones"].astype(bool, copy=True),
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
            for buffer in buffers
            for item in buffer.transitions
        )
        result.update(
            {
                "macro_actor_optimizer_steps_this_update": len(records),
                "macro_sampled_termination_decisions_per_frame_mean": float(
                    np.mean(sampled_termination_decisions)
                ),
                "macro_sampled_termination_decisions_per_frame_max": int(
                    np.max(sampled_termination_decisions)
                ),
                "macro_joint_log_ratio_abs_mean_by_duration": (
                    macro_joint_log_ratio_abs_mean_by_duration
                ),
                "macro_critic_optimizer_steps_this_update": critic_fit["optimizer_steps"],
                "macro_critic_update_iterations_this_update": critic_fit[
                    "update_iterations"
                ],
                "long_value_optimizer_steps_this_update": long_value_fit[
                    "optimizer_steps"
                ],
                "macro_actor_advantage_mode": MACRO_ACTOR_ADVANTAGE_MODE,
                "macro_critic_target_mode": self.macro_critic_target_mode,
                "macro_actor_advantage_mean": float(adv_mean),
                "macro_actor_advantage_std": float(adv_std),
                "macro_update_selected_shared_advantage_mean": float(
                    np.mean(normalized_actor_advantage[selected_groups])
                    if np.any(selected_groups)
                    else 0.0
                ),
                "macro_update_selected_shared_advantage_std": float(
                    np.std(normalized_actor_advantage[selected_groups])
                    if np.any(selected_groups)
                    else 0.0
                ),
                "macro_shared_advantage_mean": float(np.mean(normalized_actor_advantage)),
                "macro_shared_advantage_std": float(np.std(normalized_actor_advantage)),
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
                "critic_group_episode_indices": (
                    list(range(int(episode_start_index), episode_end_index + 1))
                    + ([] if last1_buffer is None else [int(last1_buffer["episode_index"])])
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
                "macro_actor_objective": "one joint PPO clipped surrogate over Cache candidate and sampled termination sequence",
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
        """按当前正式梯度组合合同拟合 raw-MSE N2 Critic。"""

        if int(steps) != 50:
            raise ValueError("Macro Critic update budget is fixed at 50 iterations.")
        if not math.isclose(float(value_loss_coefficient), 0.01, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Macro Critic loss coefficient must remain 0.01.")
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
            raise RuntimeError("N2 group targets must stay detached and frozen.")
        if current_target_tensor.requires_grad or (
            last1_target_tensor is not None and last1_target_tensor.requires_grad
        ):
            raise RuntimeError("N2 group targets must be detached.")
        if optimizer_steps != int(steps):
            raise RuntimeError("Macro Critic optimizer step count mismatch.")

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
                "actual_durations",
                "rewards",
                "old_values",
                "old_next_values",
                "next_observations",
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
        checkpoint_role: str = "RESUMABLE_MAINLINE_CHECKPOINT",
    ) -> dict[str, Any]:
        """构造只包含当前 PROPOSED identity 的可恢复 checkpoint。"""

        completed = int(self.training_counters["completed_episodes"])
        update_count = int(self.training_counters["policy_update_count"])
        if completed % self.realizations_per_policy_update != 0:
            raise ValueError("checkpoint completed_episodes is not on a complete group boundary")
        if update_count != completed // self.realizations_per_policy_update:
            raise ValueError("checkpoint policy_update_count disagrees with completed_episodes")
        last1 = self.last1_macro_transition_buffer
        return {
            "checkpoint_version": self.checkpoint_version,
            "checkpoint_role": str(checkpoint_role),
            "algorithm_id": self.algorithm_id,
            "macro_critic_target_mode": self.macro_critic_target_mode,
            "long_value_enabled": self.long_value_enabled,
            "long_value_target": self.long_value_target,
            "method": self.checkpoint_method,
            "policy_schema": self.policy.architecture_id,
            "architecture_version": int(self.policy.architecture_version),
            "config_contract": self.config_contract,
            "unified_cache_action": bool(self.unified_cache_action),
            "cardinality_corrected_cache_action": bool(
                self.cardinality_corrected_cache_action
            ),
            "initialization_metadata": dict(self.initialization_metadata),
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
            "n2_critic_state_dict": self.policy.n2_critic_state_dict(),
            "macro_critic_state_dict": self.policy.n2_critic_state_dict(),
            "long_value_critic_state_dict": self.policy.long_value_critic_state_dict(),
            "micro_optimizer_state_dict": self.micro_optimizer.state_dict(),
            "macro_optimizer_state_dict": self.macro_optimizer.state_dict(),
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
        checkpoint_role: str = "RESUMABLE_MAINLINE_CHECKPOINT",
    ) -> Path:
        destination = Path(path)
        _atomic_torch_save(
            self.checkpoint_payload(
                config_raw_sha256=config_raw_sha256,
                config_semantic_sha256=config_semantic_sha256,
                episode_pool=episode_pool,
                seed=seed,
                best_episode_return=best_episode_return,
                checkpoint_role=checkpoint_role,
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
        """只恢复 identity 与配置合同均精确匹配的 PROPOSED checkpoint。"""

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
        if payload.get("long_value_enabled") is not self.long_value_enabled:
            raise ValueError("checkpoint long_value_enabled mismatch.")
        if payload.get("long_value_target") != self.long_value_target:
            raise ValueError("checkpoint long_value_target mismatch.")
        if payload.get("algorithm_id") != self.algorithm_id:
            raise ValueError("checkpoint algorithm_id mismatch.")
        if payload.get("method") != self.checkpoint_method:
            raise ValueError("checkpoint method mismatch.")
        if payload.get("policy_schema") != self.policy.architecture_id:
            raise ValueError("checkpoint policy_schema mismatch.")
        saved_contract = payload.get("config_contract")
        if not isinstance(saved_contract, dict):
            raise ValueError("checkpoint policy tensor/configuration contract is missing.")
        if (
            saved_contract.get("algorithm_id") != self.algorithm_id
            or saved_contract.get("policy_schema") != self.policy.architecture_id
        ):
            raise ValueError("checkpoint config_contract identity mismatch.")
        required = {
            "checkpoint_version": self.checkpoint_version,
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
        if saved_contract != self.config_contract:
            raise ValueError("checkpoint policy tensor/configuration contract mismatch.")
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
        )
        self.initialization_metadata = dict(payload.get("initialization_metadata", {}))
        self.macro_optimizer.load_state_dict(payload["macro_optimizer_state_dict"])
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
        expected_update_count = completed_episodes // self.realizations_per_policy_update
        if completed_episodes % self.realizations_per_policy_update != 0:
            raise ValueError(
                "checkpoint completed_episodes is not on a policy-update group boundary"
            )
        if (
            int(self.training_counters["policy_update_count"]) != expected_update_count
            or int(self.training_counters["macro_update_count"]) != expected_update_count
            or int(self.training_counters["micro_update_count"]) != expected_update_count
            or int(self.training_counters["long_value_update_count"]) != expected_update_count
        ):
            raise ValueError("checkpoint update counters disagree with completed_episodes")
        if raw_last1 is None:
            if completed_episodes != 0:
                raise ValueError("PROPOSED checkpoint is missing its last1 Macro transition buffer.")
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
                raise ValueError("PROPOSED last1 Macro transition buffer schema mismatch.")
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
                raise ValueError("PROPOSED last1 Macro transition buffer is invalid.")
            self.last1_macro_transition_buffer = {
                "episode_index": completed_episodes,
                "observations": observations.copy(),
                "next_observations": next_observations.copy(),
                "macro_rewards": rewards.copy(),
                "actual_durations": durations.copy(),
                "dones": dones.copy(),
            }
        self._apply_macro_actor_learning_rate(
            int(self.training_counters["policy_update_count"])
        )
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
) -> dict[str, Any]:
    """按完整 episode 收集 on-policy 数据并更新 PROPOSED 主线策略。"""

    _require_torch()
    if int(target_episodes) <= 0:
        raise ValueError("--epochs 必须为正整数。")
    if save_macro_transition_trace and (resume or episode_seed_start is None):
        raise ValueError("transition-trace reporting requires a fresh run with --episode-seed-start.")
    config_file = Path(config_path).expanduser().resolve(strict=True)
    cfg = project_config.load_config(str(config_file))
    project_config.validate_for_formal_experiment(cfg)
    group_size = int(cfg["algorithm"]["ppo"]["realizations_per_policy_update"])
    if int(target_episodes) % group_size != 0:
        raise ValueError("--epochs must be divisible by realizations_per_policy_update")
    policy_update_total = int(target_episodes) // group_size
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
        policy_update_total=policy_update_total,
        capture_macro_transition_trace=save_macro_transition_trace,
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
            ("target_episodes", int(target_episodes)),
            ("realizations_per_policy_update", group_size),
            ("policy_update_total", policy_update_total),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"run manifest {field} mismatch。")
        if not isinstance(manifest.get("policy_schema"), str):
            raise ValueError("run manifest policy_schema is missing。")
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
        if completed % group_size != 0:
            raise ValueError("checkpoint completed_episodes is not on a group boundary")
        _reconcile_run_artifacts_with_checkpoint(
            run_path,
            completed,
            episode_slots=trainer.context.episode_slots,
            group_size=group_size,
            slot_start=trainer.context.train_origin_start,
        )
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
    last_episode_record: dict[str, Any] | None = None
    last_update_record: dict[str, Any] | None = None

    def save_run_summary() -> None:
        summary = {
            "experiment_type": "macro_cache_smdp_micro_uav_ppo",
            "algorithm_id": trainer.algorithm_id,
            "architecture_version": int(trainer.policy.architecture_version),
            "checkpoint_version": trainer.checkpoint_version,
            "completed_episodes": int(completed),
            "policy_update_count": int(
                trainer.training_counters["policy_update_count"]
            ),
            "policy_update_total": int(trainer.policy_update_total),
            "realizations_per_policy_update": int(
                trainer.realizations_per_policy_update
            ),
            "target_episodes": int(target_episodes),
            "best_episode_return": best_return,
            "early_stop_reason": early_stop_reason,
            "early_stop_detail": early_stop_detail,
            "training_counters": dict(trainer.training_counters),
            "initialization_metadata": dict(trainer.initialization_metadata),
            "unified_cache_action": bool(trainer.unified_cache_action),
            "cardinality_corrected_cache_action": bool(
                trainer.cardinality_corrected_cache_action
            ),
            "last_episode": last_episode_record,
            "last_update": last_update_record,
        }
        with run_summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")

    def read_last_record(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        last_line = ""
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        record = json.loads(last_line)
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} final record is not an object")
        return record

    if resume:
        last_episode_record = read_last_record(episode_metrics_path)
        last_update_record = read_last_record(update_metrics_path)
        if completed:
            update_training_epoch_return_plot(run_path)
        else:
            stale_plot = run_path / "plots" / TRAINING_EPOCH_RETURN_PLOT_FILENAME
            stale_plot.unlink(missing_ok=True)

    save_run_summary()
    if not resume:
        trainer.save_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            seed=seed,
            best_episode_return=None,
        )

    for group_zero_start in range(completed, int(target_episodes), group_size):
        group_started_at = time.perf_counter()
        collections: list[EpisodeCollection] = []
        collection_runtimes: list[float] = []
        collection_paths: list[Path] = []
        collection_failure = False
        frozen_policy_sha256 = _module_state_sha256(trainer.policy)
        optimizer_step_keys = (
            "micro_optimizer_steps",
            "macro_optimizer_steps",
            "micro_actor_optimizer_steps",
            "micro_critic_optimizer_steps",
            "macro_actor_optimizer_steps",
            "macro_critic_optimizer_steps",
            "long_value_optimizer_steps",
        )
        optimizer_steps_before = tuple(
            int(trainer.training_counters[key]) for key in optimizer_step_keys
        )
        for episode_index in range(group_zero_start, group_zero_start + group_size):
            collection_started_at = time.perf_counter()
            realization_path = episode_paths[episode_index]
            next_context = load_environment_context(
                str(config_file), realization_dir=realization_path
            )
            _assert_realization_context_compatible(context, next_context)
            trainer.context = context = next_context
            episode_seed = int(
                episode_seed_start if episode_seed_start is not None else seed
            ) + episode_index
            try:
                collection = collect_dual_timescale_episode(
                    trainer.policy,
                    context,
                    aligned_train_steps(context),
                    deterministic=False,
                    episode_seed=episode_seed,
                )
            except FloatingPointError as exc:
                early_stop_reason = "NAN_INF"
                early_stop_detail = str(exc)
                collection_failure = True
                break
            if _contains_nonfinite(collection.summary):
                early_stop_reason = "NAN_INF"
                early_stop_detail = "Rollout summary contains NaN/Inf."
                collection_failure = True
                break
            if _module_state_sha256(trainer.policy) != frozen_policy_sha256:
                raise RuntimeError("Policy parameters changed during a frozen collection group.")
            optimizer_steps_after = tuple(
                int(trainer.training_counters[key]) for key in optimizer_step_keys
            )
            if optimizer_steps_after != optimizer_steps_before:
                raise RuntimeError("An optimizer stepped during a frozen collection group.")
            collections.append(collection)
            collection_runtimes.append(time.perf_counter() - collection_started_at)
            collection_paths.append(realization_path)
        if collection_failure:
            save_run_summary()
            break
        if len(collections) != group_size:
            raise RuntimeError("policy-update group collection ended before all realizations completed")

        update_started_at = time.perf_counter()
        try:
            update_metrics = trainer.update_from_episodes(collections)
        except FloatingPointError as exc:
            early_stop_reason = "NAN_INF"
            early_stop_detail = str(exc)
            save_run_summary()
            break
        if _contains_nonfinite(update_metrics):
            early_stop_reason = "NAN_INF"
            early_stop_detail = "PPO update metrics contain NaN/Inf."
            save_run_summary()
            break
        update_runtime = time.perf_counter() - update_started_at
        group_collection_runtime = float(sum(collection_runtimes))
        group_training_runtime = time.perf_counter() - group_started_at
        completed = int(trainer.training_counters["completed_episodes"])
        group_start = int(update_metrics["policy_update_group_start_episode"])
        group_end = int(update_metrics["policy_update_group_end_episode"])
        group_id = int(update_metrics["policy_update_group_id"])
        group_seeds = list(update_metrics["policy_update_group_seeds"])

        if save_macro_transition_trace:
            if len(trainer.last_macro_transition_traces) != group_size:
                raise RuntimeError("Macro trace capture did not return one trace per realization.")
            for offset, (collection, trace) in enumerate(
                zip(collections, trainer.last_macro_transition_traces, strict=True)
            ):
                episode_number = group_start + offset
                trace_count = int(len(trace["macro_reward"]))
                expected_count = int(collection.summary["macro_frame_count"])
                trace_episode = {
                    "episode_id": episode_number,
                    "episode_seed": int(group_seeds[offset]),
                    "macro_frame_index": np.arange(trace_count, dtype=np.int64),
                    **trace,
                }
                if trace_count != expected_count:
                    raise RuntimeError(
                        f"Macro trace E{episode_number} has {trace_count} transitions; "
                        f"episode summary reports {expected_count}."
                    )
                _, episode_trace_check = build_macro_transition_trace_arrays(
                    [trace_episode],
                    gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
                    expected_macro_frame_counts={episode_number: expected_count},
                )
                if episode_trace_check["transition_count"] <= 0:
                    raise RuntimeError(f"Macro trace E{episode_number} contains no transitions.")
                trace_episode_batches.append(trace_episode)

        episode_returns = [float(item.summary["episode_return"]) for item in collections]
        group_best_return = max(episode_returns)
        save_best_train_return_checkpoint = False
        if best_return is None or group_best_return > float(best_return):
            best_return = group_best_return
            save_best_train_return_checkpoint = True

        optimizer_steps_this_update = {
            key: int(update_metrics[key])
            for key in (
                "macro_actor_optimizer_steps_this_update",
                "macro_critic_optimizer_steps_this_update",
                "long_value_optimizer_steps_this_update",
                "micro_actor_optimizer_steps_this_update",
                "micro_critic_optimizer_steps_this_update",
            )
        }
        grouped_metric_prefixes = (
            "macro_actor_advantage",
            "macro_shared_advantage",
            "macro_update_selected_shared_advantage",
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
        grouped_metric_names = {
            "macro_actor_advantage_mode",
            "macro_critic_target_mode",
            "macro_critic_fit_snapshots",
            "long_value_fit_snapshots",
            "long_value_target",
            "long_value_target_detached",
            "long_value_target_frozen_during_update",
            "long_value_target_sha256_before",
            "long_value_target_sha256_after",
            "macro_joint_approx_kl",
            "macro_joint_clip_fraction",
            "macro_sampled_termination_decisions_per_frame_mean",
            "macro_sampled_termination_decisions_per_frame_max",
            "macro_joint_log_ratio_abs_mean_by_duration",
            "macro_cache_ratio_approx_kl",
            "macro_cache_ratio_clip_fraction",
            "macro_termination_ratio_approx_kl",
            "macro_termination_ratio_clip_fraction",
            "macro_actor_learning_rate_applied",
            "joint_logprob_decomposition_max_error",
            "macro_early_stop",
            "macro_entropy",
            "micro_explained_variance",
            "micro_approx_kl",
            "micro_clip_fraction",
            "micro_entropy",
            "micro_value_rmse",
            "micro_early_stop",
        }
        group_metric_copy = {
            key: value
            for key, value in update_metrics.items()
            if key.startswith(grouped_metric_prefixes) or key in grouped_metric_names
        }
        episode_records: list[dict[str, Any]] = []
        for offset, (collection, realization_path) in enumerate(
            zip(collections, collection_paths, strict=True)
        ):
            episode_number = group_start + offset
            collection_runtime = float(collection_runtimes[offset])
            episode_record = {
                "record_type": "episode",
                "episode": episode_number,
                "episode_seed": (
                    int(episode_seed_start) + episode_number - 1
                    if episode_seed_start is not None
                    else realization_path.name
                ),
                "training_epoch_runtime_seconds": collection_runtime,
                "episode_collection_runtime_seconds": collection_runtime,
                "policy_update_group_id": group_id,
                "policy_update_group_start_episode": group_start,
                "policy_update_group_end_episode": group_end,
                "policy_update_group_size": group_size,
                "update_metric_scope": "policy_update_group",
                **collection.summary,
                **group_metric_copy,
                **optimizer_steps_this_update,
            }
            episode_records.append(episode_record)

        update_record = {
            "record_type": "update",
            "episode": group_end,
            "group_episode_indices": list(range(group_start, group_end + 1)),
            "group_seeds": group_seeds,
            "group_collection_runtime_seconds": group_collection_runtime,
            "group_training_runtime_seconds": float(group_training_runtime),
            "policy_update_runtime_seconds": float(update_runtime),
            **update_metrics,
            **optimizer_steps_this_update,
            **trainer.training_counters,
        }
        critic_diagnostics_record = {
            "record_type": "critic_diagnostics",
            "episode": group_end,
            "policy_update_group_id": group_id,
            "policy_update_group_start_episode": group_start,
            "policy_update_group_end_episode": group_end,
            "policy_update_group_size": group_size,
            **{
                key: value
                for key, value in update_metrics.items()
                if "critic" in key.lower() or key.startswith("n2_")
            },
        }
        last_episode_record = episode_records[-1]
        last_update_record = update_record

        for episode_record in episode_records:
            append_dual_timescale_record(episode_metrics_path, episode_record)
        append_dual_timescale_record(update_metrics_path, update_record)
        append_dual_timescale_record(critic_diagnostics_path, critic_diagnostics_record)
        for episode_record, collection in zip(episode_records, collections, strict=True):
            append_dual_timescale_slot_metrics(
                slot_metrics_path,
                episode=int(episode_record["episode"]),
                transitions=collection.micro.transitions,
            )
            print(
                format_training_epoch_log(
                    episode_record,
                    target_epochs=int(target_episodes),
                    macro_actor_learning_rate=float(
                        update_metrics["macro_actor_learning_rate_applied"]
                    ),
                ),
                flush=True,
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
        save_run_summary()
        update_training_epoch_return_plot(run_path)
        if save_macro_transition_trace:
            trace_summary = write_macro_transition_trace_artifact(
                run_path,
                trace_episode_batches,
                gamma=float(cfg["algorithm"]["ppo"]["gamma"]),
            )
        if verbose:
            print(json.dumps(update_record, ensure_ascii=False, sort_keys=True))
        if save_best_train_return_checkpoint:
            trainer.save_checkpoint(
                checkpoint_path_for_write(run_path, BEST_TRAIN_RETURN_CHECKPOINT_FILENAME),
                config_raw_sha256=raw_hash,
                config_semantic_sha256=semantic_hash,
                episode_pool=str(pool_path),
                seed=seed,
                best_episode_return=best_return,
                checkpoint_role="TRAINING_DIAGNOSTIC_ONLY_NOT_FINAL_SELECTION",
            )
        trainer.save_checkpoint(
            checkpoint_path,
            config_raw_sha256=raw_hash,
            config_semantic_sha256=semantic_hash,
            episode_pool=str(pool_path),
            seed=seed,
            best_episode_return=best_return,
        )
    return {
        "run_dir": str(run_path),
        "completed_episodes": completed,
        "target_episodes": int(target_episodes),
        "checkpoint": str(checkpoint_path),
        "best_episode_return": best_return,
        "macro_transition_trace": trace_summary,
        "early_stop_reason": early_stop_reason,
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
    result = run_training(
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
    return 2 if result["early_stop_reason"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
