#!/usr/bin/env python
"""Run the TRAIN-only traceable Candidate Exact-Q integrity diagnostic.

The diagnostic deliberately performs one independent Cache Critic bootstrap
warmup and one independent full TRAIN rollout.  It never invokes PPO.update().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from common import project_config
from ppo_simple.cache_credit import evaluate_cache_full_information_values
from ppo_simple.rollout import compute_gae
from ppo_simple.trainer import (
    JOINT_PROPOSED_CHECKPOINT_VERSION,
    PPOV2Trainer,
    _assert_realization_context_compatible,
    _config_hash,
    _ensure_episode_realization,
    _load_single_episode_generator,
    _run_with_isolated_rng,
    load_environment_context,
)
from scenario.realization import EPISODE_SEED_STREAMS, derive_episode_seeds, derive_namespaced_seed


WARMUP_NAMESPACE = "ppo_critic_bootstrap_warmup"
DIAGNOSTIC_NAMESPACE = "ppo_candidate_q_integrity"
Q_ATOL = 1.0e-8
Q_RTOL = 1.0e-10
SHARE_EPSILON = 1.0e-12


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _correlation(left: Any, right: Any, *, spearman: bool = False) -> float | str:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FloatingPointError("correlation input is empty, misaligned, or non-finite")
    if spearman:
        x = _rankdata_average(x)
        y = _rankdata_average(y)
    if float(np.std(x)) <= np.finfo(np.float64).eps or float(np.std(y)) <= np.finfo(np.float64).eps:
        return "NOT_MEANINGFUL"
    value = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(value):
        raise FloatingPointError("correlation result is non-finite")
    return value


def _distribution(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise FloatingPointError("distribution input is empty or non-finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _percentiles(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise FloatingPointError("percentile input is empty or non-finite")
    quantiles = np.percentile(array, [0, 1, 5, 50, 95, 99, 100])
    return {
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def _error_metrics(prediction: Any, target: Any) -> dict[str, float | str]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    expected = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(predicted) != len(expected) or not len(expected):
        raise ValueError("prediction and target lengths must be equal and non-empty")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
        raise FloatingPointError("prediction or target contains NaN/Inf")
    residual = predicted - expected
    variance = float(np.var(expected))
    near_zero_variance = variance <= 1.0e-12 * max(1.0, float(np.mean(expected) ** 2))
    explained_variance: float | str
    if near_zero_variance:
        explained_variance = "NOT_MEANINGFUL"
    else:
        explained_variance = float(1.0 - np.var(residual) / variance)
        if not math.isfinite(explained_variance):
            raise FloatingPointError("explained variance is non-finite")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "explained_variance": explained_variance,
    }


def _actor_parameter_snapshot(trainer: PPOV2Trainer) -> tuple[Any, ...]:
    return tuple(parameter.detach().clone() for parameter in trainer.policy.actor_parameters())


def _critic_snapshot(trainer: PPOV2Trainer) -> dict[str, Any]:
    buffers = dict(trainer.policy.named_buffers())
    if "popart_mean" not in buffers or "popart_std" not in buffers:
        raise RuntimeError("proposed-joint PopArt buffers are missing")
    statistics = trainer.critic_return_statistics
    snapshot = {
        "digest": str(trainer._critic_parameter_digest()),
        "steps": int(trainer.critic_optimizer_steps),
        "popart_mean": float(buffers["popart_mean"].detach().cpu().item()),
        "popart_std": float(buffers["popart_std"].detach().cpu().item()),
        "return_stat_count": int(statistics.count),
        "return_stat_mean": float(statistics.mean),
        "return_stat_m2": float(statistics.m2),
    }
    if not all(math.isfinite(float(snapshot[name])) for name in (
        "popart_mean", "popart_std", "return_stat_mean", "return_stat_m2"
    )):
        raise FloatingPointError("critic or return-stat snapshot contains NaN/Inf")
    return snapshot


def _assert_critic_frozen(before: dict[str, Any], after: dict[str, Any], label: str) -> None:
    if before != after:
        raise AssertionError(f"Critic state changed during frozen phase {label}: {before!r} != {after!r}")


def _critic_values(trainer: PPOV2Trainer, observations: Any) -> np.ndarray:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        values = trainer.policy.critic_value(tensor)
    result = values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if len(result) != len(observations) or not np.all(np.isfinite(result)):
        raise FloatingPointError("critic value evaluation returned invalid values")
    return result


def _reward_to_go(rewards: Any, dones: Any, gamma: float) -> np.ndarray:
    reward_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
    done_array = np.asarray(dones, dtype=bool).reshape(-1)
    if len(reward_array) != len(done_array) or not len(reward_array):
        raise ValueError("reward-to-go inputs must be equal and non-empty")
    if not np.all(np.isfinite(reward_array)):
        raise FloatingPointError("rewards contain NaN/Inf")
    result = np.zeros_like(reward_array)
    running = 0.0
    for index in range(len(reward_array) - 1, -1, -1):
        if bool(done_array[index]):
            running = 0.0
        running = float(reward_array[index]) + float(gamma) * running
        result[index] = running
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reward-to-go contains NaN/Inf")
    return result


def _select_diagnostic_indices(dones: Any) -> tuple[int, ...]:
    nonterminal = np.flatnonzero(~np.asarray(dones, dtype=bool).reshape(-1))
    if len(nonterminal) < 3:
        raise ValueError("at least three nonterminal transitions are required")
    positions = [int(round((len(nonterminal) - 1) * fraction)) for fraction in (0.10, 0.50, 0.90)]
    selected = tuple(int(nonterminal[position]) for position in positions)
    if len(set(selected)) != 3:
        raise AssertionError("position-based selection did not produce three unique transitions")
    return selected


def _role_name(value: str) -> str:
    return str(value).strip().upper().split("_", 1)[0]


def _role_metrics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("a Cache role has no exact Candidate records")
    delta_reward = np.asarray([row["delta_reward"] for row in candidates], dtype=np.float64)
    gamma_delta_v = np.asarray([row["gamma_delta_v"] for row in candidates], dtype=np.float64)
    delta_q = np.asarray([row["delta_q"] for row in candidates], dtype=np.float64)
    denominator = np.abs(delta_reward) + np.abs(gamma_delta_v) + SHARE_EPSILON
    reward_share = np.abs(delta_reward) / denominator
    critic_share = np.abs(gamma_delta_v) / denominator
    if not all(np.all(np.isfinite(values)) for values in (delta_reward, gamma_delta_v, delta_q, reward_share, critic_share)):
        raise FloatingPointError("role decomposition contains NaN/Inf")
    return {
        "candidate_count": int(len(candidates)),
        "abs_delta_reward": _distribution(np.abs(delta_reward)),
        "abs_gamma_delta_v": _distribution(np.abs(gamma_delta_v)),
        "abs_delta_q": _distribution(np.abs(delta_q)),
        "pearson_delta_q_vs_delta_reward": _correlation(delta_q, delta_reward),
        "spearman_delta_q_vs_delta_reward": _correlation(delta_q, delta_reward, spearman=True),
        "pearson_delta_q_vs_gamma_delta_v": _correlation(delta_q, gamma_delta_v),
        "spearman_delta_q_vs_gamma_delta_v": _correlation(delta_q, gamma_delta_v, spearman=True),
        "reward_abs_share": _distribution(reward_share),
        "critic_abs_share": _distribution(critic_share),
    }


def _transition_head_diagnostics(
    trainer: PPOV2Trainer,
    selected_indices: tuple[int, ...],
    q_result: dict[str, Any],
    gamma: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    q_components = q_result.get("q_component_diagnostics")
    full_components = q_result.get("full_candidate_q_components")
    if not isinstance(q_components, list) or len(q_components) != len(selected_indices):
        raise AssertionError("Cache KEEP component diagnostics are incomplete")
    if not isinstance(full_components, list):
        raise AssertionError("full Candidate component diagnostics are missing")
    keep_by_key: dict[tuple[int, int], dict[str, float]] = {}
    head_summaries: list[dict[str, Any]] = []
    for sample_index, transition_rows in enumerate(q_components):
        transition_index = selected_indices[sample_index]
        if len(transition_rows) != len(trainer.context.action_spec.cache_node_ids):
            raise AssertionError("Cache component diagnostics do not cover every head")
        for head in transition_rows:
            key = (transition_index, int(head["head_index"]))
            keep = {name: float(head["keep"][name]) for name in ("reward", "next_v", "q")}
            if not all(math.isfinite(value) for value in keep.values()):
                raise FloatingPointError("KEEP Exact-Q components are non-finite")
            reconstructed = keep["reward"] + float(gamma) * keep["next_v"]
            if not math.isclose(keep["q"], reconstructed, rel_tol=Q_RTOL, abs_tol=Q_ATOL):
                raise AssertionError(f"KEEP Q identity failed at transition/head {key}")
            keep_by_key[key] = keep

    candidates: list[dict[str, Any]] = []
    max_q_error = 0.0
    max_delta_error = 0.0
    seen_head_keys: set[tuple[int, int]] = set()
    for full_row in full_components:
        transition_index = int(full_row["transition_index"])
        head_index = int(full_row["head_index"])
        key = (transition_index, head_index)
        seen_head_keys.add(key)
        if key not in keep_by_key:
            raise AssertionError(f"full Candidate row lacks its KEEP reference: {key}")
        keep = keep_by_key[key]
        head_id = str(trainer.context.action_spec.cache_node_ids[head_index])
        role = _role_name(str(trainer.context.cache_nodes["node_roles"][head_index]))
        for raw in full_row["candidates"]:
            reward = float(raw["reward"])
            next_v = float(raw["next_v"])
            q_value = float(raw["q"])
            if not all(math.isfinite(value) for value in (reward, next_v, q_value)):
                raise FloatingPointError("full-exact Candidate record contains NaN/Inf")
            expected_q = reward + float(gamma) * next_v
            q_error = abs(q_value - expected_q)
            if not math.isclose(q_value, expected_q, rel_tol=Q_RTOL, abs_tol=Q_ATOL):
                raise AssertionError(f"Candidate Q identity failed at {key}, index={raw['universe_index']}")
            delta_reward = reward - keep["reward"]
            delta_next_v = next_v - keep["next_v"]
            gamma_delta_v = float(gamma) * delta_next_v
            delta_q = q_value - keep["q"]
            reconstructed_delta_q = delta_reward + gamma_delta_v
            delta_error = abs(delta_q - reconstructed_delta_q)
            if not math.isclose(delta_q, reconstructed_delta_q, rel_tol=Q_RTOL, abs_tol=Q_ATOL):
                raise AssertionError(f"delta-Q identity failed at {key}, index={raw['universe_index']}")
            max_q_error = max(max_q_error, q_error, abs(keep["q"] - (keep["reward"] + gamma * keep["next_v"])))
            max_delta_error = max(max_delta_error, delta_error)
            candidates.append({
                "transition_index": transition_index,
                "origin_index": int(full_row["origin_index"]),
                "head_index": head_index,
                "head_id": head_id,
                "role": role,
                "universe_index": int(raw["universe_index"]),
                "target_node_cache": [int(value) for value in raw["target_node_cache"]],
                "reward": reward,
                "next_v": next_v,
                "q": q_value,
                "keep_reward": keep["reward"],
                "keep_next_v": keep["next_v"],
                "keep_q": keep["q"],
                "delta_reward": delta_reward,
                "delta_next_v": delta_next_v,
                "gamma_delta_v": gamma_delta_v,
                "delta_q": delta_q,
                "reconstructed_delta_q": reconstructed_delta_q,
                "q_reconstruction_error": q_error,
                "delta_q_error": delta_error,
                "reward_abs_share": abs(delta_reward) / (abs(delta_reward) + abs(gamma_delta_v) + SHARE_EPSILON),
                "critic_abs_share": abs(gamma_delta_v) / (abs(delta_reward) + abs(gamma_delta_v) + SHARE_EPSILON),
            })

    expected_candidate_rows = int(q_result["logical_candidate_count"])
    if len(candidates) != expected_candidate_rows:
        raise AssertionError(f"full Candidate components count mismatch: {len(candidates)} != {expected_candidate_rows}")
    if int(q_result["exact_candidate_count"]) != expected_candidate_rows:
        raise AssertionError("full-exact evaluator did not exactly evaluate every legal Candidate")
    if int(q_result["surrogate_candidate_count"]) != 0:
        raise AssertionError("surrogate Q values were used in a forced full-exact run")
    candidate_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in candidates:
        candidate_groups.setdefault((item["transition_index"], item["head_index"]), []).append(item)
    for key, rows in candidate_groups.items():
        ordered = sorted(rows, key=lambda item: (int(item["universe_index"]),))
        winner = max(ordered, key=lambda item: (float(item["q"]), -int(item["universe_index"])))
        head_summaries.append({
            "transition_index": key[0],
            "head_index": key[1],
            "head_id": winner["head_id"],
            "role": winner["role"],
            "legal_candidate_count": len(ordered),
            "keep": {
                "reward": winner["keep_reward"],
                "next_v": winner["keep_next_v"],
                "q": winner["keep_q"],
            },
            "true_exact_best": {
                "universe_index": winner["universe_index"],
                "target_node_cache": winner["target_node_cache"],
                "reward": winner["reward"],
                "next_v": winner["next_v"],
                "q": winner["q"],
                "delta_reward": winner["delta_reward"],
                "gamma_delta_v": winner["gamma_delta_v"],
                "delta_q": winner["delta_q"],
                "reward_abs_share": winner["reward_abs_share"],
                "critic_abs_share": winner["critic_abs_share"],
            },
            "candidate_q_min": min(item["q"] for item in ordered),
            "candidate_q_max": max(item["q"] for item in ordered),
            "candidate_q_mean": float(np.mean([item["q"] for item in ordered])),
        })
    if len(head_summaries) != len(seen_head_keys):
        raise AssertionError("candidate head summaries are inconsistent")
    return candidates, head_summaries, max_q_error, max_delta_error


def _critic_warmup_classification(pre: dict[str, Any], post: dict[str, Any]) -> str:
    directions: list[int] = []
    for key in ("mae", "rmse"):
        before = float(pre[key])
        after = float(post[key])
        directions.append(1 if after < before else (-1 if after > before else 0))
    pre_ev, post_ev = pre["explained_variance"], post["explained_variance"]
    if isinstance(pre_ev, (int, float)) and isinstance(post_ev, (int, float)):
        directions.append(1 if float(post_ev) > float(pre_ev) else (-1 if float(post_ev) < float(pre_ev) else 0))
    if directions and all(value == 1 for value in directions):
        return "IMPROVES"
    if directions and all(value == -1 for value in directions):
        return "DEGRADES"
    if directions and all(value == 0 for value in directions):
        return "NO_CLEAR_IMPROVEMENT"
    return "INCONCLUSIVE"


def _validate_context(context: Any) -> None:
    if context.policy_schema != "adaptive_joint_cache_uav_ppo":
        raise ValueError(f"expected proposed-joint policy schema, got {context.policy_schema!r}")
    if context.data_protocol != project_config.PPO_DATA_PROTOCOL_FULL_EPISODE_RANDOMIZED:
        raise ValueError("diagnostic requires full_episode_randomized TRAIN protocol")
    if int(context.episode_slots) != int(context.configured_rollout_steps):
        raise ValueError("configured rollout must cover the complete TRAIN episode")


class _ScratchDirectory:
    """Delete scratch only on success; retain it when any phase fails."""

    def __init__(self, parent: Path) -> None:
        self.parent = parent.resolve(strict=True)
        self.path: Path | None = None

    def __enter__(self) -> str:
        self.path = Path(tempfile.mkdtemp(prefix="traceable_candidate_q_integrity_", dir=self.parent))
        return str(self.path)

    def __exit__(self, exc_type: Any, exc: Any, _traceback: Any) -> bool:
        if self.path is None:
            return False
        if exc_type is not None:
            print(f"SCRATCH_PRESERVED_FOR_RECOVERY={self.path}", file=sys.stderr, flush=True)
            return False
        resolved = self.path.resolve(strict=True)
        prefix = str(self.parent).rstrip("\\/") + os.sep
        if not str(resolved).startswith(prefix):
            raise RuntimeError(f"refusing to remove scratch outside temp directory: {resolved}")
        shutil.rmtree(resolved)
        if resolved.exists():
            raise RuntimeError(f"scratch cleanup failed: {resolved}")
        print("TRACEABLE_DIAGNOSTIC_SCRATCH_CLEANED=YES", file=sys.stderr, flush=True)
        return False


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(serialized, encoding="utf-8")
    reread = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reread, dict):
        raise TypeError(f"persisted JSON must contain an object: {path}")
    return reread


def _build_cache_result_section(
    *,
    q_components: list[Any],
    cache_q_result: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    head_summaries: list[dict[str, Any]],
    max_q_error: float,
    max_delta_q_error: float,
    by_role: dict[str, Any],
    uav_focus: list[dict[str, Any]],
    uav_top10: list[dict[str, Any]],
    actual_state_distribution: dict[str, float],
    counterfactual_distribution: dict[str, float],
    role_support: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete Cache-Q report section from named inputs."""
    if not q_components or not all(isinstance(rows, list) for rows in q_components):
        raise AssertionError("Cache Q component rows are empty or malformed")
    return {
        "full_exact_used": "YES",
        "surrogate_training_executed": "NO",
        "surrogate_q_used_for_diagnostic": "NO",
        "cache_head_count": int(len(q_components[0])),
        "transition_head_evaluation_count": int(sum(len(rows) for rows in q_components)),
        "candidate_count": int(len(candidate_rows)),
        "exact_candidate_count": int(cache_q_result["exact_candidate_count"]),
        "surrogate_candidate_count": int(cache_q_result["surrogate_candidate_count"]),
        "logical_candidate_count": int(cache_q_result["logical_candidate_count"]),
        "keep_branch_count": int(cache_q_result["keep_branch_count"]),
        "environment_execution_count": int(cache_q_result["counterfactual_environment_executions"]),
        "q_identity_atol": Q_ATOL,
        "q_identity_rtol": Q_RTOL,
        "max_q_reconstruction_error": float(max_q_error),
        "max_delta_q_reconstruction_error": float(max_delta_q_error),
        "all_q_identities_passed": True,
        "by_role": by_role,
        "head_summaries": head_summaries,
        "uav_0_uav_1_exact_best": uav_focus,
        "uav_0_uav_1_top10": uav_top10,
        "actual_state_critic_value": actual_state_distribution,
        "counterfactual_candidate_next_v": counterfactual_distribution,
        "counterfactual_support_by_role": role_support,
        "all_candidate_records": candidate_rows,
    }


def _assemble_final_result(nested: dict[str, Any]) -> dict[str, Any]:
    """Add stable flat gate fields to the fully populated diagnostic record."""
    provenance = nested["provenance"]
    warmup = nested["warmup"]
    diagnostic = nested["diagnostic_rollout"]
    freeze = nested["critic_freeze"]
    cache_q = nested["cache_q"]
    classification = nested["classification"]
    pre_mc = diagnostic["pre_warmup_critic_vs_mc_return"]
    post_mc = diagnostic["post_warmup_critic_vs_mc_return"]
    pre_gae = diagnostic["pre_warmup_critic_vs_gae_return"]
    post_gae = diagnostic["post_warmup_critic_vs_gae_return"]
    gae_mc = diagnostic["gae_return_vs_mc_return"]
    result = dict(nested)
    result.update({
        "CURRENT_HEAD": provenance["current_head"],
        "GIT_DIRTY_BEFORE": provenance["git_dirty_before"],
        "ARCHITECTURE_VERSION": int(provenance["architecture_version"]),
        "CHECKPOINT_VERSION": int(provenance["checkpoint_version"]),
        "WARMUP_REALIZATION_SEED": int(provenance["warmup_realization_seed"]),
        "DIAGNOSTIC_REALIZATION_SEED": int(provenance["diagnostic_realization_seed"]),
        "REAL_WARMUP_TRANSITIONS": int(warmup["rollout_transition_count"]),
        "REAL_WARMUP_CRITIC_STEPS": int(warmup["critic_steps"]),
        "REAL_WARMUP_ACTOR_PARAM_DELTA": float(warmup["actor_parameter_delta"]),
        "DIAGNOSTIC_TRANSITIONS": int(diagnostic["transition_count"]),
        "DIAGNOSTIC_TRANSITION_INDICES": list(diagnostic["selected_transition_indices"]),
        "DIAGNOSTIC_ORIGIN_INDICES": list(diagnostic["selected_origin_indices"]),
        "PRE_WARMUP_CRITIC_MAE_TO_MC_RETURN": pre_mc["mae"],
        "POST_WARMUP_CRITIC_MAE_TO_MC_RETURN": post_mc["mae"],
        "PRE_WARMUP_CRITIC_RMSE_TO_MC_RETURN": pre_mc["rmse"],
        "POST_WARMUP_CRITIC_RMSE_TO_MC_RETURN": post_mc["rmse"],
        "PRE_WARMUP_CRITIC_EV_TO_MC_RETURN": pre_mc["explained_variance"],
        "POST_WARMUP_CRITIC_EV_TO_MC_RETURN": post_mc["explained_variance"],
        "PRE_WARMUP_CRITIC_MAE_TO_GAE_RETURN": pre_gae["mae"],
        "POST_WARMUP_CRITIC_MAE_TO_GAE_RETURN": post_gae["mae"],
        "GAE_RETURN_VS_MC_RETURN_MAE": gae_mc["mae"],
        "GAE_RETURN_VS_MC_RETURN_RMSE": gae_mc["rmse"],
        "GAE_RETURN_VS_MC_RETURN_CORRELATION": gae_mc["correlation"],
        "CRITIC_WARMUP_REAL_DATA_CLASSIFICATION": warmup["classification"],
        "CRITIC_PARAMETER_DIGEST": provenance["critic_parameter_sha256"],
        "CRITIC_OPTIMIZER_STEPS": int(provenance["critic_optimizer_steps"]),
        "POPART_MEAN": float(provenance["popart_mean"]),
        "POPART_STD": float(provenance["popart_std"]),
        "RETURN_STAT_COUNT": int(provenance["return_stat_count"]),
        "CRITIC_DIGEST_BEFORE_Q_DIAGNOSTIC": freeze["before_q_diagnostic"]["digest"],
        "CRITIC_DIGEST_AFTER_Q_DIAGNOSTIC": freeze["after_q_diagnostic"]["digest"],
        "CRITIC_STEPS_BEFORE_Q_DIAGNOSTIC": int(freeze["before_q_diagnostic"]["steps"]),
        "CRITIC_STEPS_AFTER_Q_DIAGNOSTIC": int(freeze["after_q_diagnostic"]["steps"]),
        "CRITIC_FROZEN_DURING_Q_DIAGNOSTIC": freeze["frozen_during_post_warmup_diagnostics"],
        "MAX_Q_RECONSTRUCTION_ERROR": float(cache_q["max_q_reconstruction_error"]),
        "MAX_DELTA_Q_RECONSTRUCTION_ERROR": float(cache_q["max_delta_q_reconstruction_error"]),
        "ROLE_DECOMPOSITION": cache_q["by_role"],
        "ACTUAL_STATE_CRITIC_P01": cache_q["actual_state_critic_value"]["p01"],
        "ACTUAL_STATE_CRITIC_P50": cache_q["actual_state_critic_value"]["p50"],
        "ACTUAL_STATE_CRITIC_P99": cache_q["actual_state_critic_value"]["p99"],
        "COUNTERFACTUAL_NEXT_V_P01": cache_q["counterfactual_candidate_next_v"]["p01"],
        "COUNTERFACTUAL_NEXT_V_P50": cache_q["counterfactual_candidate_next_v"]["p50"],
        "COUNTERFACTUAL_NEXT_V_P99": cache_q["counterfactual_candidate_next_v"]["p99"],
        "COUNTERFACTUAL_SUPPORT_BY_ROLE": cache_q["counterfactual_support_by_role"],
        "UAV_0_UAV_1_EXACT_BEST": cache_q["uav_0_uav_1_exact_best"],
        "UAV_0_UAV_1_TOP10": cache_q["uav_0_uav_1_top10"],
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION": classification["candidate_q_label_integrity"],
        "COUNTERFACTUAL_CRITIC_RELIABILITY": classification["counterfactual_critic_reliability"],
        "NEXT_STAGE": classification["next_stage"],
        "FORMAL_PPO_UPDATE_EXECUTED": classification["formal_ppo_update_executed"],
        "ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC": int(classification["actor_optimizer_steps_during_diagnostic"]),
        "ONE_EPOCH_TRAINING_EXECUTED": classification["one_epoch_training_executed"],
        "READY_FOR_FORMAL_TRAINING_STAGE": classification["ready_for_formal_training_stage"],
    })
    return result


def _validate_final_result(result: dict[str, Any]) -> None:
    required = (
        "CURRENT_HEAD", "ARCHITECTURE_VERSION", "CHECKPOINT_VERSION",
        "WARMUP_REALIZATION_SEED", "DIAGNOSTIC_REALIZATION_SEED",
        "REAL_WARMUP_TRANSITIONS", "REAL_WARMUP_CRITIC_STEPS", "REAL_WARMUP_ACTOR_PARAM_DELTA",
        "PRE_WARMUP_CRITIC_MAE_TO_MC_RETURN", "POST_WARMUP_CRITIC_MAE_TO_MC_RETURN",
        "PRE_WARMUP_CRITIC_RMSE_TO_MC_RETURN", "POST_WARMUP_CRITIC_RMSE_TO_MC_RETURN",
        "PRE_WARMUP_CRITIC_EV_TO_MC_RETURN", "POST_WARMUP_CRITIC_EV_TO_MC_RETURN",
        "PRE_WARMUP_CRITIC_MAE_TO_GAE_RETURN", "POST_WARMUP_CRITIC_MAE_TO_GAE_RETURN",
        "GAE_RETURN_VS_MC_RETURN_MAE", "GAE_RETURN_VS_MC_RETURN_RMSE",
        "GAE_RETURN_VS_MC_RETURN_CORRELATION", "CRITIC_WARMUP_REAL_DATA_CLASSIFICATION",
        "CRITIC_PARAMETER_DIGEST", "CRITIC_OPTIMIZER_STEPS", "POPART_MEAN", "POPART_STD",
        "RETURN_STAT_COUNT", "CRITIC_DIGEST_BEFORE_Q_DIAGNOSTIC", "CRITIC_DIGEST_AFTER_Q_DIAGNOSTIC",
        "CRITIC_STEPS_BEFORE_Q_DIAGNOSTIC", "CRITIC_STEPS_AFTER_Q_DIAGNOSTIC",
        "CRITIC_FROZEN_DURING_Q_DIAGNOSTIC", "MAX_Q_RECONSTRUCTION_ERROR",
        "MAX_DELTA_Q_RECONSTRUCTION_ERROR", "ROLE_DECOMPOSITION", "ACTUAL_STATE_CRITIC_P01",
        "ACTUAL_STATE_CRITIC_P50", "ACTUAL_STATE_CRITIC_P99", "COUNTERFACTUAL_NEXT_V_P01",
        "COUNTERFACTUAL_NEXT_V_P50", "COUNTERFACTUAL_NEXT_V_P99", "COUNTERFACTUAL_SUPPORT_BY_ROLE",
        "UAV_0_UAV_1_EXACT_BEST", "UAV_0_UAV_1_TOP10",
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION", "COUNTERFACTUAL_CRITIC_RELIABILITY", "NEXT_STAGE",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"final diagnostic JSON is missing required fields: {missing}")
    if result["DIAGNOSTIC_TRANSITION_INDICES"] != [30, 149, 268] or result["DIAGNOSTIC_ORIGIN_INDICES"] != [30, 149, 268]:
        raise AssertionError("diagnostic indices differ from the authorized fixed selection")
    if result["FORMAL_PPO_UPDATE_EXECUTED"] != "NO" or int(result["ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC"]) != 0:
        raise AssertionError("final result violates the no-formal-update contract")
    if result["ONE_EPOCH_TRAINING_EXECUTED"] != "NO" or result["READY_FOR_FORMAL_TRAINING_STAGE"] != "NO":
        raise AssertionError("final result violates the no-formal-training gate")
    if int(result["REAL_WARMUP_CRITIC_STEPS"]) != 40 or float(result["REAL_WARMUP_ACTOR_PARAM_DELTA"]) != 0.0:
        raise AssertionError("final result violates the critic-only warmup contract")
    if int(result["DIAGNOSTIC_REALIZATION_SEED"]) != 12184115623283823756 or int(result["WARMUP_REALIZATION_SEED"]) != 7868781415413077432:
        raise AssertionError("final result does not use the authorized fixed realization seeds")
    def check_finite(value: Any, location: str = "root") -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"non-finite result field: {location}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_finite(item, f"{location}[{index}]")
    check_finite(result)


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.10g}"


def _render_traceable_section(result: dict[str, Any]) -> str:
    _validate_final_result(result)
    cache = result["cache_q"]
    diag = result["diagnostic_rollout"]
    warmup = result["warmup"]
    roles = cache["by_role"]
    lines = [
        "### Authorized compensatory rerun",
        "",
        "```text",
        "AUTHORIZED_RERUN = YES",
        "RERUN_REASON = RESULT_ASSEMBLY_BUG_AFTER_SUCCESSFUL_COMPUTATION",
        f"CURRENT_HEAD = {result['CURRENT_HEAD']}",
        f"GIT_DIRTY_BEFORE = {result['GIT_DIRTY_BEFORE']}",
        f"ARCHITECTURE_VERSION = {result['ARCHITECTURE_VERSION']}",
        f"CHECKPOINT_VERSION = {result['CHECKPOINT_VERSION']}",
        f"WARMUP_REALIZATION_SEED = {result['WARMUP_REALIZATION_SEED']}",
        f"DIAGNOSTIC_REALIZATION_SEED = {result['DIAGNOSTIC_REALIZATION_SEED']}",
        f"FORMAL_EPISODE_SEED_STREAM_CHANGED = {result['provenance']['formal_episode_seed_stream_changed']}",
        f"REAL_WARMUP_TRANSITIONS = {result['REAL_WARMUP_TRANSITIONS']}",
        f"REAL_WARMUP_CRITIC_STEPS = {result['REAL_WARMUP_CRITIC_STEPS']}",
        f"REAL_WARMUP_ACTOR_PARAM_DELTA = {_fmt(result['REAL_WARMUP_ACTOR_PARAM_DELTA'])}",
        f"DIAGNOSTIC_TRANSITIONS = {result['DIAGNOSTIC_TRANSITIONS']}",
        f"DIAGNOSTIC_TRANSITION_INDICES = {result['DIAGNOSTIC_TRANSITION_INDICES']}",
        f"DIAGNOSTIC_ORIGIN_INDICES = {result['DIAGNOSTIC_ORIGIN_INDICES']}",
        f"PRE_WARMUP_CRITIC_MAE_TO_MC_RETURN = {_fmt(result['PRE_WARMUP_CRITIC_MAE_TO_MC_RETURN'])}",
        f"POST_WARMUP_CRITIC_MAE_TO_MC_RETURN = {_fmt(result['POST_WARMUP_CRITIC_MAE_TO_MC_RETURN'])}",
        f"PRE_WARMUP_CRITIC_RMSE_TO_MC_RETURN = {_fmt(result['PRE_WARMUP_CRITIC_RMSE_TO_MC_RETURN'])}",
        f"POST_WARMUP_CRITIC_RMSE_TO_MC_RETURN = {_fmt(result['POST_WARMUP_CRITIC_RMSE_TO_MC_RETURN'])}",
        f"PRE_WARMUP_CRITIC_EV_TO_MC_RETURN = {_fmt(result['PRE_WARMUP_CRITIC_EV_TO_MC_RETURN'])}",
        f"POST_WARMUP_CRITIC_EV_TO_MC_RETURN = {_fmt(result['POST_WARMUP_CRITIC_EV_TO_MC_RETURN'])}",
        f"PRE_WARMUP_CRITIC_MAE_TO_GAE_RETURN = {_fmt(result['PRE_WARMUP_CRITIC_MAE_TO_GAE_RETURN'])}",
        f"POST_WARMUP_CRITIC_MAE_TO_GAE_RETURN = {_fmt(result['POST_WARMUP_CRITIC_MAE_TO_GAE_RETURN'])}",
        f"GAE_RETURN_VS_MC_RETURN_MAE = {_fmt(result['GAE_RETURN_VS_MC_RETURN_MAE'])}",
        f"GAE_RETURN_VS_MC_RETURN_RMSE = {_fmt(result['GAE_RETURN_VS_MC_RETURN_RMSE'])}",
        f"GAE_RETURN_VS_MC_RETURN_CORRELATION = {_fmt(result['GAE_RETURN_VS_MC_RETURN_CORRELATION'])}",
        f"CRITIC_WARMUP_REAL_DATA_CLASSIFICATION = {result['CRITIC_WARMUP_REAL_DATA_CLASSIFICATION']}",
        f"CRITIC_PARAMETER_DIGEST = {result['CRITIC_PARAMETER_DIGEST']}",
        f"CRITIC_OPTIMIZER_STEPS = {result['CRITIC_OPTIMIZER_STEPS']}",
        f"POPART_MEAN = {_fmt(result['POPART_MEAN'])}",
        f"POPART_STD = {_fmt(result['POPART_STD'])}",
        f"RETURN_STAT_COUNT = {result['RETURN_STAT_COUNT']}",
        f"CRITIC_DIGEST_BEFORE_Q_DIAGNOSTIC = {result['CRITIC_DIGEST_BEFORE_Q_DIAGNOSTIC']}",
        f"CRITIC_DIGEST_AFTER_Q_DIAGNOSTIC = {result['CRITIC_DIGEST_AFTER_Q_DIAGNOSTIC']}",
        f"CRITIC_STEPS_BEFORE_Q_DIAGNOSTIC = {result['CRITIC_STEPS_BEFORE_Q_DIAGNOSTIC']}",
        f"CRITIC_STEPS_AFTER_Q_DIAGNOSTIC = {result['CRITIC_STEPS_AFTER_Q_DIAGNOSTIC']}",
        f"CRITIC_FROZEN_DURING_Q_DIAGNOSTIC = {result['CRITIC_FROZEN_DURING_Q_DIAGNOSTIC']}",
        f"FULL_EXACT_USED = {cache['full_exact_used']}",
        f"SURROGATE_TRAINING_EXECUTED = {cache['surrogate_training_executed']}",
        f"SURROGATE_Q_USED_FOR_DIAGNOSTIC = {cache['surrogate_q_used_for_diagnostic']}",
        f"FULL_EXACT_CACHE_HEAD_COUNT = {cache['cache_head_count']}",
        f"FULL_EXACT_CANDIDATE_COUNT = {cache['candidate_count']}",
        f"MAX_Q_RECONSTRUCTION_ERROR = {_fmt(result['MAX_Q_RECONSTRUCTION_ERROR'])}",
        f"MAX_DELTA_Q_RECONSTRUCTION_ERROR = {_fmt(result['MAX_DELTA_Q_RECONSTRUCTION_ERROR'])}",
    ]
    for role in ("SV", "RSU", "UAV"):
        stats = roles[role]
        lines.extend([
            f"{role}_DELTA_REWARD_ABS_MEDIAN = {_fmt(stats['abs_delta_reward']['median'])}",
            f"{role}_GAMMA_DELTA_V_ABS_MEDIAN = {_fmt(stats['abs_gamma_delta_v']['median'])}",
            f"{role}_DELTA_Q_ABS_MEDIAN = {_fmt(stats['abs_delta_q']['median'])}",
            f"{role}_REWARD_ABS_SHARE_MEDIAN = {_fmt(stats['reward_abs_share']['median'])}",
            f"{role}_CRITIC_ABS_SHARE_MEDIAN = {_fmt(stats['critic_abs_share']['median'])}",
        ])
    lines.extend([
        f"ACTUAL_STATE_CRITIC_P01 = {_fmt(result['ACTUAL_STATE_CRITIC_P01'])}",
        f"ACTUAL_STATE_CRITIC_P50 = {_fmt(result['ACTUAL_STATE_CRITIC_P50'])}",
        f"ACTUAL_STATE_CRITIC_P99 = {_fmt(result['ACTUAL_STATE_CRITIC_P99'])}",
        f"COUNTERFACTUAL_NEXT_V_P01 = {_fmt(result['COUNTERFACTUAL_NEXT_V_P01'])}",
        f"COUNTERFACTUAL_NEXT_V_P50 = {_fmt(result['COUNTERFACTUAL_NEXT_V_P50'])}",
        f"COUNTERFACTUAL_NEXT_V_P99 = {_fmt(result['COUNTERFACTUAL_NEXT_V_P99'])}",
    ])
    for role in ("SV", "RSU", "UAV"):
        stats = cache["counterfactual_support_by_role"][role]
        lines.append(f"{role}_COUNTERFACTUAL_BELOW_ACTUAL_P01_FRACTION = {_fmt(stats['below_actual_p01_fraction'])}")
        lines.append(f"{role}_COUNTERFACTUAL_ABOVE_ACTUAL_P99_FRACTION = {_fmt(stats['above_actual_p99_fraction'])}")
    for target_id in ("UAV_0", "UAV_1"):
        rows = [row for row in cache["uav_0_uav_1_exact_best"] if row["head_id"] == target_id]
        max_q = max(abs(float(row["true_exact_best"]["delta_q"])) for row in rows)
        max_reward = max(abs(float(row["true_exact_best"]["delta_reward"])) for row in rows)
        max_critic = max(abs(float(row["true_exact_best"]["gamma_delta_v"])) for row in rows)
        lines.extend([
            f"{target_id}_MAX_DELTA_Q = {_fmt(max_q)}",
            f"{target_id}_MAX_DELTA_REWARD = {_fmt(max_reward)}",
            f"{target_id}_MAX_GAMMA_DELTA_V = {_fmt(max_critic)}",
        ])
    lines.extend([
        f"CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = {result['CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION']}",
        f"COUNTERFACTUAL_CRITIC_RELIABILITY = {result['COUNTERFACTUAL_CRITIC_RELIABILITY']}",
        f"FORMAL_PPO_UPDATE_EXECUTED = {result['FORMAL_PPO_UPDATE_EXECUTED']}",
        f"ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC = {result['ACTOR_OPTIMIZER_STEPS_DURING_DIAGNOSTIC']}",
        f"ONE_EPOCH_TRAINING_EXECUTED = {result['ONE_EPOCH_TRAINING_EXECUTED']}",
        f"READY_FOR_FORMAL_TRAINING_STAGE = {result['READY_FOR_FORMAL_TRAINING_STAGE']}",
        f"NEXT_STAGE = {result['NEXT_STAGE']}",
        "```",
        "",
        "Critic value metrics use the same 300 diagnostic observations before and after warmup. GAE uses the saved zero-step rollout values, configured gamma, and configured lambda; Monte Carlo reward-to-go ends at the terminal transition.",
        "",
        "| Role | Candidates | |Δreward| median / p90 / p95 / max | |γΔV| median / p90 / p95 / max | |ΔQ| median / p90 / p95 / max | Reward share mean / median / p90 | Critic share mean / median / p90 | Pearson/Spearman ΔQ vs reward | Pearson/Spearman ΔQ vs γΔV |",
        "|---|---:|---|---|---|---|---|---|",
    ])
    for role in ("SV", "RSU", "UAV"):
        stats = roles[role]
        def dist(key: str) -> str:
            d = stats[key]
            return " / ".join(_fmt(d[name]) for name in ("median", "p90", "p95", "max"))
        reward_share = stats["reward_abs_share"]
        critic_share = stats["critic_abs_share"]
        lines.append(
            f"| {role} | {stats['candidate_count']} | {dist('abs_delta_reward')} | {dist('abs_gamma_delta_v')} | {dist('abs_delta_q')} | "
            f"{' / '.join(_fmt(reward_share[n]) for n in ('mean','median','p90'))} | "
            f"{' / '.join(_fmt(critic_share[n]) for n in ('mean','median','p90'))} | "
            f"{_fmt(stats['pearson_delta_q_vs_delta_reward'])} / {_fmt(stats['spearman_delta_q_vs_delta_reward'])} | "
            f"{_fmt(stats['pearson_delta_q_vs_gamma_delta_v'])} / {_fmt(stats['spearman_delta_q_vs_gamma_delta_v'])} |"
        )
    lines.extend([
        "",
        "| Role | Candidate next-V min / p01 / p05 / p50 / p95 / p99 / max | Below actual P01 | Above actual P99 |",
        "|---|---|---:|---:|",
    ])
    for role in ("SV", "RSU", "UAV"):
        stats = cache["counterfactual_support_by_role"][role]
        p = stats["candidate_next_v"]
        values = (p["min"],p["p01"],p["p05"],p["p50"],p["p95"],p["p99"],p["max"])
        lines.append(f"| {role} | {' / '.join(_fmt(x) for x in values)} | {_fmt(stats['below_actual_p01_fraction'])} | {_fmt(stats['above_actual_p99_fraction'])} |")
    lines.extend([
        "",
        "| Transition | Head | Legal Candidates | KEEP reward / next-V / Q | Exact best config | Exact best reward / next-V / Q | Δreward / γΔV / ΔQ | Reward / critic share |",
        "|---:|---|---:|---|---|---|---|---|",
    ])
    for row in cache["head_summaries"]:
        keep, best = row["keep"], row["true_exact_best"]
        lines.append(
            f"| {row['transition_index']} | {row['head_id']} | {row['legal_candidate_count']} | "
            f"{_fmt(keep['reward'])} / {_fmt(keep['next_v'])} / {_fmt(keep['q'])} | `{best['target_node_cache']}` | "
            f"{_fmt(best['reward'])} / {_fmt(best['next_v'])} / {_fmt(best['q'])} | "
            f"{_fmt(best['delta_reward'])} / {_fmt(best['gamma_delta_v'])} / {_fmt(best['delta_q'])} | "
            f"{_fmt(best['reward_abs_share'])} / {_fmt(best['critic_abs_share'])} |"
        )
    lines.extend([
        "",
        "| Transition | UAV head | Rank | Config | Reward | next-V | Q | Δreward | γΔV | ΔQ |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cache["uav_0_uav_1_top10"]:
        lines.append(
            f"| {row['transition_index']} | {row['head_id']} | {row['rank']} | `{row['config']}` | "
            f"{_fmt(row['reward'])} | {_fmt(row['next_v'])} | {_fmt(row['q'])} | "
            f"{_fmt(row['delta_reward'])} | {_fmt(row['gamma_delta_v'])} | {_fmt(row['delta_q'])} |"
        )
    lines.extend([
        "",
        "Every legal Candidate configuration and its reward / next-V / Q decomposition is retained in [`traceable_candidate_q_integrity_result.json`](traceable_candidate_q_integrity_result.json). The Q and delta-Q checks used `atol=1e-8`, `rtol=1e-10`; all recorded values were finite.",
        "",
        "The exact algebra establishes component consistency only. Candidate-Q classification also considers real-data warmup behavior and counterfactual next-V support.",
        "",
    ])
    return "\n".join(lines)


def _apply_rerun_report(existing: str, result: dict[str, Any]) -> str:
    heading = "## Traceable Candidate-Q Integrity Gate"
    if heading not in existing:
        raise AssertionError("second-stage report lacks the original Candidate-Q gate heading")
    if "### Authorized compensatory rerun" in existing:
        raise AssertionError("authorized rerun report subsection already exists")
    old_paragraph = (
        "No second real warmup or rollout was started because this gate explicitly\n"
        "limits the run to one of each. Numeric pre/post critic metrics, per-role\n"
        "decomposition summaries, UAV_0/UAV_1 rankings, and the temporary checkpoint\n"
        "hash therefore remain unavailable. The gate stays inconclusive until a\n"
        "separately authorized rerun can persist the complete report."
    )
    if old_paragraph not in existing:
        raise AssertionError("first-attempt provenance paragraph was not found verbatim")
    amended = existing.replace(
        old_paragraph,
        "At the time of the first-attempt report, no retry had started. The user later authorized one compensatory rerun, documented below.",
        1,
    )
    return amended.rstrip() + "\n\n" + _render_traceable_section(result)


def _apply_one_epoch_report(existing: str, result: dict[str, Any]) -> str:
    old_completion_paragraph = (
        "The one-epoch run was not executed. The proposed-joint candidate critic\n"
        "cold-start repair is implemented. One traceable Candidate-Q diagnostic run\n"
        "completed its TRAIN-only warmup and exact evaluation, but a result-assembly\n"
        "error prevented its numeric evidence from being persisted. No formal epoch\n"
        "was run."
    )
    completion_paragraph = (
        "The one-epoch run was not executed. The proposed-joint candidate critic\n"
        "cold-start repair is implemented. The authorized compensatory traceable\n"
        "Candidate-Q diagnostic completed its TRAIN-only warmup and exact evaluation,\n"
        "and its complete numeric evidence is persisted in the accompanying [result\n"
        "JSON](traceable_candidate_q_integrity_result.json). No formal PPO epoch\n"
        "was run."
    )
    if old_completion_paragraph in existing:
        existing = existing.replace(old_completion_paragraph, completion_paragraph, 1)
    elif completion_paragraph not in existing:
        raise AssertionError("one-epoch report completion summary is neither stale nor current")

    reason = (
        "ONE_EPOCH_NOT_EXECUTED_REASON = Traceable Candidate-Q diagnostics completed; "
        "formal PPO epoch remains deferred to the next stage"
    )
    import re
    amended, count = re.subn(
        r"^ONE_EPOCH_NOT_EXECUTED_REASON = .*?$",
        reason,
        existing,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise AssertionError("one-epoch report reason field was not found exactly once")
    amended, ready_count = re.subn(
        r"^READY_FOR_FORMAL_TRAINING_STAGE = (?:YES|NO)$",
        f"READY_FOR_FORMAL_TRAINING_STAGE = {result['READY_FOR_FORMAL_TRAINING_STAGE']}",
        amended,
        count=1,
        flags=re.MULTILINE,
    )
    if ready_count != 1:
        raise AssertionError("one-epoch readiness field was not found exactly once")
    for required_line in (
        "ONE_EPOCH_TRAINING_EXECUTED = NO",
        "ONE_EPOCH_COMPLETED_EPOCHS = 0",
        "EPOCH_2_STARTED = NO",
    ):
        if required_line not in amended:
            raise AssertionError(f"one-epoch invariant missing after report update: {required_line}")
    return amended


def _validate_updated_reports(second_stage_text: str, one_epoch_text: str) -> None:
    for required in (
        "AUTHORIZED_RERUN = YES",
        "RERUN_REASON = RESULT_ASSEMBLY_BUG_AFTER_SUCCESSFUL_COMPUTATION",
        "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION =",
        "COUNTERFACTUAL_CRITIC_RELIABILITY =",
        "NEXT_STAGE =",
        "UAV_0",
        "UAV_1",
        "FULL_EXACT_USED = YES",
    ):
        if required not in second_stage_text:
            raise AssertionError(f"updated second-stage report missing {required!r}")
    for required in (
        "ONE_EPOCH_TRAINING_EXECUTED = NO",
        "ONE_EPOCH_COMPLETED_EPOCHS = 0",
        "EPOCH_2_STARTED = NO",
        "READY_FOR_FORMAL_TRAINING_STAGE = NO",
        "traceable_candidate_q_integrity_result.json",
    ):
        if required not in one_epoch_text:
            raise AssertionError(f"updated one-epoch report missing {required!r}")


def _synthetic_assembly_smoke() -> None:
    """Exercise cache-result assembly, final fields, Markdown access and JSON I/O."""
    d = {"mean": 0.2, "median": 0.2, "p90": 0.3, "p95": 0.4, "max": 0.5}
    role = {
        "candidate_count": 1,
        "abs_delta_reward": d,
        "abs_gamma_delta_v": d,
        "abs_delta_q": d,
        "pearson_delta_q_vs_delta_reward": 1.0,
        "spearman_delta_q_vs_delta_reward": 1.0,
        "pearson_delta_q_vs_gamma_delta_v": 1.0,
        "spearman_delta_q_vs_gamma_delta_v": 1.0,
        "reward_abs_share": d,
        "critic_abs_share": d,
    }
    support_role = {
        "candidate_count": 1,
        "candidate_next_v": {"min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        "below_actual_p01_fraction": 0.0,
        "above_actual_p99_fraction": 0.0,
    }
    candidate = {
        "transition_index": 30, "origin_index": 30, "head_index": 0, "head_id": "UAV_0", "role": "UAV",
        "universe_index": 0, "target_node_cache": [1], "reward": 1.0, "next_v": 0.0, "q": 1.0,
        "keep_reward": 0.0, "keep_next_v": 0.0, "keep_q": 0.0, "delta_reward": 1.0,
        "delta_next_v": 0.0, "gamma_delta_v": 0.0, "delta_q": 1.0, "reconstructed_delta_q": 1.0,
        "q_reconstruction_error": 0.0, "delta_q_error": 0.0, "reward_abs_share": 1.0, "critic_abs_share": 0.0,
    }
    best = {
        "transition_index": 30, "origin_index": 30, "head_id": "UAV_0", "legal_candidate_count": 1,
        "keep": {"reward": 0.0, "next_v": 0.0, "q": 0.0},
        "true_exact_best": {"config": [1], "reward": 1.0, "next_v": 0.0, "q": 1.0,
                            "delta_reward": 1.0, "gamma_delta_v": 0.0, "delta_q": 1.0,
                            "reward_abs_share": 1.0, "critic_abs_share": 0.0},
    }
    top = {"transition_index": 30, "origin_index": 30, "head_id": "UAV_0", "rank": 1,
           "config": [1], "reward": 1.0, "next_v": 0.0, "q": 1.0,
           "delta_reward": 1.0, "gamma_delta_v": 0.0, "delta_q": 1.0}
    distribution = {"min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    cache = _build_cache_result_section(
        q_components=[[{}, {}, {}]],
        cache_q_result={"exact_candidate_count": 1, "surrogate_candidate_count": 0,
                        "logical_candidate_count": 1, "keep_branch_count": 3,
                        "counterfactual_environment_executions": 4},
        candidate_rows=[candidate], head_summaries=[{
            "transition_index": 30, "head_id": "UAV_0", "legal_candidate_count": 1,
            "keep": {"reward": 0.0, "next_v": 0.0, "q": 0.0},
            "true_exact_best": {"target_node_cache": [1], **best["true_exact_best"]},
        }],
        max_q_error=0.0, max_delta_q_error=0.0,
        by_role={"SV": role, "RSU": role, "UAV": role},
        uav_focus=[best] * 6, uav_top10=[top],
        actual_state_distribution=distribution,
        counterfactual_distribution=distribution,
        role_support={"SV": support_role, "RSU": support_role, "UAV": support_role},
    )
    q_metrics = {
        **cache,
        "cache_head_count": 3,
    }
    error_metrics = {"mae": 0.1, "rmse": 0.2, "explained_variance": 0.3}
    best_uav1 = dict(best, head_id="UAV_1")
    nested = {
        "provenance": {"current_head": "head", "git_dirty_before": "YES", "architecture_version": 6,
                       "checkpoint_version": 26, "warmup_realization_seed": 7868781415413077432,
                       "diagnostic_realization_seed": 12184115623283823756, "critic_parameter_sha256": "c" * 64,
                       "critic_optimizer_steps": 40, "popart_mean": 0.0, "popart_std": 1.0,
                       "return_stat_count": 300, "formal_episode_seed_stream_changed": "NO"},
        "warmup": {"rollout_transition_count": 300, "critic_steps": 40, "actor_parameter_delta": 0.0,
                   "classification": "IMPROVES"},
        "diagnostic_rollout": {"transition_count": 300, "selected_transition_indices": [30, 149, 268],
            "selected_origin_indices": [30, 149, 268], "pre_warmup_critic_vs_mc_return": error_metrics,
            "post_warmup_critic_vs_mc_return": error_metrics, "pre_warmup_critic_vs_gae_return": error_metrics,
            "post_warmup_critic_vs_gae_return": error_metrics,
            "gae_return_vs_mc_return": {"mae": 0.1, "rmse": 0.2, "correlation": 0.9}},
        "critic_freeze": {"before_q_diagnostic": {"digest": "a" * 64, "steps": 40},
            "after_q_diagnostic": {"digest": "a" * 64, "steps": 40},
            "frozen_during_post_warmup_diagnostics": "YES"},
        "cache_q": q_metrics,
        "classification": {"candidate_q_label_integrity": "CANDIDATE_Q_INTEGRITY_SUPPORTED",
            "counterfactual_critic_reliability": "SUPPORTED", "next_stage": "CANDIDATE_SURROGATE_SCREENING_REVALIDATION",
            "formal_ppo_update_executed": "NO", "actor_optimizer_steps_during_diagnostic": 0,
            "one_epoch_training_executed": "NO", "ready_for_formal_training_stage": "NO"},
    }
    nested["critic_freeze"].update({"digest_equal": True, "optimizer_steps_equal": True,
        "popart_equal": True, "return_statistics_equal": True})
    nested["cache_q"]["uav_0_uav_1_exact_best"] = [best] * 3 + [best_uav1] * 3
    nested["cache_q"].update({"actual_state_critic_value": distribution,
        "counterfactual_candidate_next_v": distribution, "counterfactual_support_by_role": cache["counterfactual_support_by_role"]})
    final = _assemble_final_result(nested)
    _validate_final_result(final)
    rendered = _render_traceable_section(final)
    one_epoch = _apply_one_epoch_report(
        "The one-epoch run was not executed. The proposed-joint candidate critic\n"
        "cold-start repair is implemented. One traceable Candidate-Q diagnostic run\n"
        "completed its TRAIN-only warmup and exact evaluation, but a result-assembly\n"
        "error prevented its numeric evidence from being persisted. No formal epoch\n"
        "was run.\n\nONE_EPOCH_NOT_EXECUTED_REASON = old\nONE_EPOCH_TRAINING_EXECUTED = NO\n"
        "ONE_EPOCH_COMPLETED_EPOCHS = 0\nEPOCH_2_STARTED = NO\nREADY_FOR_FORMAL_TRAINING_STAGE = NO\n",
        final,
    )
    if (
        "UAV_0" not in rendered
        or "MAX_DELTA_Q" not in rendered
        or "READY_FOR_FORMAL_TRAINING_STAGE = NO" not in one_epoch
        or "complete numeric evidence is persisted" not in one_epoch
    ):
        raise AssertionError("synthetic report-field smoke failed")
    second_stage_before = (
        "## Traceable Candidate-Q Integrity Gate\n\n"
        "No second real warmup or rollout was started because this gate explicitly\n"
        "limits the run to one of each. Numeric pre/post critic metrics, per-role\n"
        "decomposition summaries, UAV_0/UAV_1 rankings, and the temporary checkpoint\n"
        "hash therefore remain unavailable. The gate stays inconclusive until a\n"
        "separately authorized rerun can persist the complete report."
    )
    synthetic_second_stage_after = _apply_rerun_report(second_stage_before, final)
    _validate_updated_reports(synthetic_second_stage_after, one_epoch)
    with tempfile.TemporaryDirectory(prefix="traceable_q_assembly_smoke_") as temp:
        path = Path(temp) / "smoke.json"
        _write_json(path, final)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        _validate_final_result(loaded)
        if len(json.dumps(loaded, ensure_ascii=False, allow_nan=False)) == 0:
            raise AssertionError("synthetic final JSON is empty")


def _run_diagnostic(
    config_path: str,
    *,
    device: str,
    validate_only: bool,
    output_json: Path,
) -> dict[str, Any] | None:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = ROOT / config_file
    config_file = config_file.resolve(strict=True)
    current_head = _git("rev-parse", "HEAD")
    git_status_before = _git("status", "--short")
    dirty_before = bool(git_status_before)
    if not current_head:
        raise RuntimeError("git HEAD provenance is empty")
    output_json = output_json.resolve(strict=False)
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite an existing formal diagnostic artifact: {output_json}")

    base_context = load_environment_context(str(config_file))
    _validate_context(base_context)
    base_seed = int(project_config.get_config_value(base_context.cfg, "experiment.seed"))
    formal_seeds_before = derive_episode_seeds(base_seed)
    gamma = float(project_config.get_config_value(base_context.cfg, "algorithm.ppo.gamma"))
    gae_lambda = float(project_config.get_config_value(base_context.cfg, "algorithm.ppo.gae_lambda"))
    if not (math.isfinite(gamma) and math.isfinite(gae_lambda)):
        raise FloatingPointError("gamma or GAE lambda is non-finite")

    warmup_seed = derive_namespaced_seed(base_seed, WARMUP_NAMESPACE, "episode_realization")
    diagnostic_seed = derive_namespaced_seed(base_seed, DIAGNOSTIC_NAMESPACE, "episode_realization")
    if warmup_seed == diagnostic_seed:
        raise AssertionError("independent realization namespaces collided")
    if warmup_seed != 7868781415413077432 or diagnostic_seed != 12184115623283823756:
        raise AssertionError("fixed authorized rerun seeds do not match the failed run")

    if validate_only:
        trainer = _run_with_isolated_rng(
            derive_namespaced_seed(base_seed, DIAGNOSTIC_NAMESPACE, "trainer_initialization"),
            lambda: PPOV2Trainer(base_context, device=device, exact_q_parallel_workers=1),
        )
        if not trainer._requires_critic_bootstrap_teacher_warmup():
            raise RuntimeError("diagnostic trainer does not use the critic bootstrap warmup lifecycle")
        return None

    diagnostic_generation_seed = derive_namespaced_seed(diagnostic_seed, DIAGNOSTIC_NAMESPACE, "episode_generation")
    diagnostic_collection_seed = derive_namespaced_seed(diagnostic_seed, DIAGNOSTIC_NAMESPACE, "rollout_collection")
    warmup_generation_seed = derive_namespaced_seed(warmup_seed, WARMUP_NAMESPACE, "episode_generation")
    warmup_collection_seed = derive_namespaced_seed(warmup_seed, WARMUP_NAMESPACE, "rollout_collection")
    warmup_optimizer_seed = derive_namespaced_seed(warmup_seed, WARMUP_NAMESPACE, "critic_optimizer")
    initialization_seed = derive_namespaced_seed(base_seed, DIAGNOSTIC_NAMESPACE, "trainer_initialization")

    scratch_parent = Path(tempfile.gettempdir())
    with _ScratchDirectory(scratch_parent) as scratch:
        scratch_root = Path(scratch)
        diagnostic_dir = scratch_root / "diagnostic_train_realization"
        warmup_dir = scratch_root / "warmup_train_realization"
        generator = _load_single_episode_generator(ROOT)

        trainer = _run_with_isolated_rng(
            initialization_seed,
            lambda: PPOV2Trainer(base_context, device=device, exact_q_parallel_workers=1),
        )
        if not trainer._requires_critic_bootstrap_teacher_warmup():
            raise RuntimeError("fresh trainer is not a proposed-joint warmup trainer")
        if trainer.actor_optimizer_steps != 0 or trainer.critic_optimizer_steps != 0:
            raise AssertionError("fresh trainer has nonzero optimizer steps")
        trainer._validate_critic_warmup_lifecycle(0, actor_optimizer_steps=0)

        print("Generating the independent diagnostic TRAIN realization first.", file=sys.stderr, flush=True)
        _run_with_isolated_rng(
            diagnostic_generation_seed,
            lambda: _ensure_episode_realization(
                base_context,
                str(config_file),
                diagnostic_dir,
                diagnostic_seed,
                generate_missing=True,
                generation_device=device,
                generator_module=generator,
            ),
        )
        diagnostic_context = load_environment_context(str(config_file), realization_dir=diagnostic_dir)
        _assert_realization_context_compatible(base_context, diagnostic_context)
        _validate_context(diagnostic_context)
        trainer.context = diagnostic_context
        diagnostic_rollout = _run_with_isolated_rng(
            diagnostic_collection_seed,
            lambda: trainer.collect_chunk(
                diagnostic_context.train_origin_start,
                diagnostic_context.initial_cache,
                diagnostic_context.initial_uav,
                diagnostic_context.configured_rollout_steps,
            ),
        )
        if len(diagnostic_rollout.transitions) != diagnostic_context.episode_slots:
            raise AssertionError("diagnostic rollout did not cover the complete TRAIN episode")
        if not bool(diagnostic_rollout.dones[-1]) or bool(np.any(diagnostic_rollout.dones[:-1])):
            raise AssertionError("diagnostic TRAIN rollout must terminate only at its final transition")
        diagnostic_indices = _select_diagnostic_indices(diagnostic_rollout.dones)
        diagnostic_origin_indices = [int(diagnostic_rollout.transitions[index].origin_index) for index in diagnostic_indices]
        if diagnostic_indices != (30, 149, 268) or diagnostic_origin_indices != [30, 149, 268]:
            raise AssertionError("fixed position selection differs from the authorized transition/origin indices")
        pre_values = _critic_values(trainer, diagnostic_rollout.observations)
        old_values = np.asarray(diagnostic_rollout.old_values, dtype=np.float64).reshape(-1)
        if old_values.shape != pre_values.shape or not np.allclose(old_values, pre_values, rtol=1.0e-6, atol=1.0e-7):
            raise AssertionError("diagnostic rollout old_values do not match zero-step critic values")
        diagnostic_phase = {
            "rollout_executed": "YES",
            "transition_count": int(len(diagnostic_rollout.transitions)),
            "selected_transition_indices": list(diagnostic_indices),
            "selected_origin_indices": diagnostic_origin_indices,
            "selected_position_fractions": [0.10, 0.50, 0.90],
            "rewards": np.asarray(diagnostic_rollout.rewards, dtype=np.float64).tolist(),
            "dones": np.asarray(diagnostic_rollout.dones, dtype=bool).tolist(),
            "observations": np.asarray(diagnostic_rollout.observations, dtype=np.float32).tolist(),
            "zero_step_old_values": old_values.tolist(),
            "zero_step_critic_values": pre_values.tolist(),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
        }
        _write_json(scratch_root / "diagnostic_rollout_metrics.json", diagnostic_phase)

        print("Diagnostic TRAIN rollout collected; now generating the independent warmup realization.", file=sys.stderr, flush=True)
        _run_with_isolated_rng(
            warmup_generation_seed,
            lambda: _ensure_episode_realization(
                base_context,
                str(config_file),
                warmup_dir,
                warmup_seed,
                generate_missing=True,
                generation_device=device,
                generator_module=generator,
            ),
        )
        warmup_context = load_environment_context(str(config_file), realization_dir=warmup_dir)
        _assert_realization_context_compatible(base_context, warmup_context)
        _validate_context(warmup_context)
        trainer.context = warmup_context
        warmup_rollout = _run_with_isolated_rng(
            warmup_collection_seed,
            lambda: trainer.collect_chunk(
                warmup_context.train_origin_start,
                warmup_context.initial_cache,
                warmup_context.initial_uav,
                warmup_context.configured_rollout_steps,
            ),
        )
        if len(warmup_rollout.transitions) != warmup_context.episode_slots:
            raise AssertionError("warmup rollout did not cover the complete TRAIN episode")
        if not bool(warmup_rollout.dones[-1]) or bool(np.any(warmup_rollout.dones[:-1])):
            raise AssertionError("warmup TRAIN rollout must terminate only at its final transition")

        actor_before_warmup = _actor_parameter_snapshot(trainer)
        trainer.critic_warmup_seed = int(warmup_seed)
        warmup_metrics = trainer.warmup_critic(warmup_rollout, rng_seed=warmup_optimizer_seed)
        actor_after_warmup = _actor_parameter_snapshot(trainer)
        actor_delta = trainer._parameter_delta_norm(actor_before_warmup, actor_after_warmup)
        if actor_delta != 0.0 or int(trainer.actor_optimizer_steps) != 0 or int(trainer.optimizer_steps) != 0:
            raise AssertionError("Actor parameters or optimizer steps changed during Critic warmup")
        if not bool(warmup_metrics["warmup_complete"]):
            raise AssertionError("real Critic warmup did not complete")
        if int(warmup_metrics["executed_critic_optimizer_steps"]) != 40:
            raise AssertionError("real Critic warmup did not execute the configured 40 steps")
        if int(trainer.critic_optimizer_steps) != 40 or int(trainer.critic_warmup_rollout_slots) != len(warmup_rollout.transitions):
            raise AssertionError("Critic warmup lifecycle counters do not match the real rollout")
        if int(trainer.cache_surrogate_training_count) != 0:
            raise AssertionError("Cache surrogate training occurred during Critic warmup")
        if int(warmup_metrics["cache_full_information_teacher_calls"]) != 0 or int(warmup_metrics["uav_exact_teacher_calls"]) != 0:
            raise AssertionError("warmup unexpectedly generated full-information teachers")

        warmup_phase = {
            "warmup_rollout_executed": "YES",
            "rollout_transition_count": int(len(warmup_rollout.transitions)),
            "critic_steps": int(warmup_metrics["executed_critic_optimizer_steps"]),
            "actor_parameter_delta": float(actor_delta),
            "actor_optimizer_steps": int(trainer.actor_optimizer_steps),
            "critic_parameter_delta_norm": float(warmup_metrics["critic_parameter_delta_norm"]),
            "warmup_metrics": warmup_metrics,
            "realization_seed": int(warmup_seed),
            "critic_optimizer_rng_seed": int(warmup_optimizer_seed),
            "critic_snapshot_after_warmup": _critic_snapshot(trainer),
        }
        _write_json(scratch_root / "warmup_metrics.json", warmup_phase)

        trainer.context = diagnostic_context
        post_warmup_snapshot = _critic_snapshot(trainer)
        post_values = _critic_values(trainer, diagnostic_rollout.observations)
        _assert_critic_frozen(post_warmup_snapshot, _critic_snapshot(trainer), "post-warmup value evaluation")

        rewards = np.asarray(diagnostic_rollout.rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(diagnostic_rollout.dones, dtype=bool).reshape(-1)
        if len(rewards) != len(pre_values) or not np.all(np.isfinite(rewards)):
            raise FloatingPointError("diagnostic rewards do not align with the saved observations")
        mc_returns = _reward_to_go(rewards, dones, gamma)
        _, gae_returns_f32 = compute_gae(
            rewards,
            pre_values,
            dones,
            0.0,
            gamma,
            gae_lambda,
        )
        gae_returns = np.asarray(gae_returns_f32, dtype=np.float64)
        if not bool(dones[-1]):
            raise AssertionError("complete diagnostic episode must have a terminal final transition")

        pre_mc = _error_metrics(pre_values, mc_returns)
        post_mc = _error_metrics(post_values, mc_returns)
        pre_gae = _error_metrics(pre_values, gae_returns)
        post_gae = _error_metrics(post_values, gae_returns)
        gae_mc = _error_metrics(gae_returns, mc_returns)
        gae_mc_correlation = _correlation(gae_returns, mc_returns)
        warmup_classification = _critic_warmup_classification(pre_mc, post_mc)
        critic_phase = {
            "pre_warmup_critic_values": pre_values.tolist(),
            "post_warmup_critic_values": post_values.tolist(),
            "mc_reward_to_go": mc_returns.tolist(),
            "diagnostic_gae_returns": gae_returns.tolist(),
            "pre_critic_vs_mc": pre_mc,
            "post_critic_vs_mc": post_mc,
            "pre_critic_vs_gae": pre_gae,
            "post_critic_vs_gae": post_gae,
            "gae_vs_mc": {"mae": gae_mc["mae"], "rmse": gae_mc["rmse"], "correlation": gae_mc_correlation},
            "actual_state_critic_values_pre_percentiles": _percentiles(pre_values),
            "actual_state_critic_values_post_percentiles": _percentiles(post_values),
            "critic_snapshot_after_warmup": post_warmup_snapshot,
            "critic_snapshot_after_post_value_evaluation": _critic_snapshot(trainer),
            "classification": warmup_classification,
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
        }
        _write_json(scratch_root / "critic_pre_post_metrics.json", critic_phase)

        # Protect the forbidden formal PPO update path for the remainder of this run.
        def forbidden_formal_update(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("trainer.update() is forbidden in this diagnostic")

        trainer.update = forbidden_formal_update  # type: ignore[method-assign]
        actor_after_warmup = _actor_parameter_snapshot(trainer)
        critic_before_q = _critic_snapshot(trainer)
        pre_q_surrogate_training_count = int(trainer.cache_surrogate_training_count)

        config_digest = _config_hash(str(config_file))
        checkpoint_path = scratch_root / "warmup_provenance_checkpoint.pt"
        checkpoint_payload = trainer.checkpoint_payload(
            0,
            int(diagnostic_context.train_origin_start),
            diagnostic_context.initial_cache,
            diagnostic_context.initial_uav,
            config_digest,
            int(diagnostic_context.configured_rollout_steps),
            episode_seed_start=None,
        )
        torch.save(checkpoint_payload, checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint_metadata = {
            "checkpoint_version": int(checkpoint_payload["checkpoint_version"]),
            "architecture_version": int(checkpoint_payload["architecture_version"]),
            "policy_schema": str(checkpoint_payload["policy_schema"]),
            "completed_training_episodes": int(checkpoint_payload["completed_training_episodes"]),
            "critic_bootstrap_teacher_warmup_complete": bool(checkpoint_payload["critic_bootstrap_teacher_warmup_complete"]),
            "critic_optimizer_steps": int(checkpoint_payload["critic_optimizer_steps"]),
            "critic_warmup_rollout_slots": int(checkpoint_payload["critic_warmup_rollout_slots"]),
            "critic_return_scale_count": int(checkpoint_payload["critic_return_scale_count"]),
            "config_sha256": str(checkpoint_payload["config_sha256"]),
        }
        if int(checkpoint_metadata["checkpoint_version"]) != JOINT_PROPOSED_CHECKPOINT_VERSION:
            raise AssertionError("temporary warmup checkpoint version is not the proposed v26 contract")
        warmup_phase.update({
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_metadata": checkpoint_metadata,
            "critic_parameter_sha256": critic_before_q["digest"],
            "critic_optimizer_steps": critic_before_q["steps"],
            "popart_mean": critic_before_q["popart_mean"],
            "popart_std": critic_before_q["popart_std"],
            "return_stat_count": critic_before_q["return_stat_count"],
        })
        _write_json(scratch_root / "warmup_metrics.json", warmup_phase)

        print(
            f"Running forced full-exact Cache Q on transition indices {diagnostic_indices}; "
            "this evaluates every legal Candidate for every Cache head.",
            file=sys.stderr,
            flush=True,
        )
        with torch.no_grad():
            cache_q_result = evaluate_cache_full_information_values(
                trainer,
                diagnostic_rollout,
                target_transition_indices=diagnostic_indices,
                force_full_exact=True,
                train_surrogate=False,
                include_q_component_diagnostics=True,
                include_full_candidate_q_components=True,
            )
        critic_after_q = _critic_snapshot(trainer)
        _assert_critic_frozen(critic_before_q, critic_after_q, "full-exact Candidate-Q evaluation")
        _assert_critic_frozen(post_warmup_snapshot, critic_after_q, "entire post-warmup diagnostic phase")
        if trainer._parameter_delta_norm(actor_after_warmup, _actor_parameter_snapshot(trainer)) != 0.0:
            raise AssertionError("Candidate-Q diagnostics changed Actor parameters")
        if int(trainer.actor_optimizer_steps) != 0 or int(trainer.optimizer_steps) != 0:
            raise AssertionError("Actor/PPO optimizer steps occurred during the diagnostic")
        if int(trainer.cache_surrogate_training_count) != pre_q_surrogate_training_count or pre_q_surrogate_training_count != 0:
            raise AssertionError("Cache surrogate training occurred during Candidate-Q diagnosis")
        if int(cache_q_result["surrogate_candidate_count"]) != 0 or int(cache_q_result["surrogate_training"]["count"]) != 0:
            raise AssertionError("Cache evaluator used or trained surrogate candidates")
        if str(cache_q_result["value_evaluator"]) != "FULL_EXACT":
            raise AssertionError("Cache Q diagnostic did not report FULL_EXACT")

        candidate_rows, head_summaries, max_q_error, max_delta_q_error = _transition_head_diagnostics(
            trainer, diagnostic_indices, cache_q_result, gamma
        )
        q_component_rows = cache_q_result.get("q_component_diagnostics")
        if not isinstance(q_component_rows, list) or len(q_component_rows) != len(diagnostic_indices):
            raise AssertionError("full-exact Cache head diagnostics were not retained")
        candidate_q_phase = {
            "full_exact_used": "YES",
            "surrogate_training_executed": "NO",
            "surrogate_q_used_for_diagnostic": "NO",
            "selected_transition_indices": list(diagnostic_indices),
            "q_component_diagnostics": q_component_rows,
            "all_candidate_records": candidate_rows,
            "head_summaries": head_summaries,
            "exact_candidate_count": int(cache_q_result["exact_candidate_count"]),
            "logical_candidate_count": int(cache_q_result["logical_candidate_count"]),
            "surrogate_candidate_count": int(cache_q_result["surrogate_candidate_count"]),
            "cache_head_count": int(len(q_component_rows[0])) if q_component_rows else 0,
            "max_q_reconstruction_error": float(max_q_error),
            "max_delta_q_reconstruction_error": float(max_delta_q_error),
            "critic_before_q_diagnostic": critic_before_q,
            "critic_after_q_diagnostic": critic_after_q,
        }
        _write_json(scratch_root / "candidate_q_components.json", candidate_q_phase)
        actual_state_distribution = _percentiles(post_values)
        candidate_next_v = np.asarray([item["next_v"] for item in candidate_rows], dtype=np.float64)
        counterfactual_distribution = _percentiles(candidate_next_v)
        actual_p01, actual_p99 = actual_state_distribution["p01"], actual_state_distribution["p99"]
        role_support: dict[str, Any] = {}
        for role in ("SV", "RSU", "UAV"):
            role_values = np.asarray([item["next_v"] for item in candidate_rows if item["role"] == role], dtype=np.float64)
            if not len(role_values):
                raise AssertionError(f"no full-exact Cache candidates were observed for role {role}")
            role_support[role] = {
                "candidate_count": int(len(role_values)),
                "candidate_next_v": _percentiles(role_values),
                "below_actual_p01_fraction": float(np.mean(role_values < actual_p01)),
                "above_actual_p99_fraction": float(np.mean(role_values > actual_p99)),
            }
        if not all(math.isfinite(float(value)) for item in role_support.values() for value in (
            item["below_actual_p01_fraction"], item["above_actual_p99_fraction"]
        )):
            raise FloatingPointError("counterfactual support fractions are non-finite")

        by_role = {
            role: _role_metrics([item for item in candidate_rows if item["role"] == role])
            for role in ("SV", "RSU", "UAV")
        }
        uav_focus: list[dict[str, Any]] = []
        uav_top10: list[dict[str, Any]] = []
        for transition_index in diagnostic_indices:
            for requested_id in ("UAV_0", "UAV_1"):
                rows = [
                    item for item in candidate_rows
                    if item["transition_index"] == transition_index
                    and item["head_id"].upper() == requested_id
                ]
                if not rows:
                    raise AssertionError(f"no exact candidates found for {requested_id} at transition {transition_index}")
                winner = max(rows, key=lambda item: (float(item["q"]), -int(item["universe_index"])))
                uav_focus.append({
                    "transition_index": transition_index,
                    "origin_index": winner["origin_index"],
                    "head_id": requested_id,
                    "legal_candidate_count": len(rows),
                    "keep": {
                        "reward": winner["keep_reward"],
                        "next_v": winner["keep_next_v"],
                        "q": winner["keep_q"],
                    },
                    "true_exact_best": {
                        "config": winner["target_node_cache"],
                        "reward": winner["reward"],
                        "next_v": winner["next_v"],
                        "q": winner["q"],
                        "delta_reward": winner["delta_reward"],
                        "gamma_delta_v": winner["gamma_delta_v"],
                        "delta_q": winner["delta_q"],
                        "reward_abs_share": winner["reward_abs_share"],
                        "critic_abs_share": winner["critic_abs_share"],
                    },
                })
                ordered = sorted(rows, key=lambda item: (-float(item["q"]), int(item["universe_index"])))[:10]
                for rank, item in enumerate(ordered, start=1):
                    uav_top10.append({
                        "transition_index": transition_index,
                        "origin_index": item["origin_index"],
                        "head_id": requested_id,
                        "rank": rank,
                        "config": item["target_node_cache"],
                        "reward": item["reward"],
                        "next_v": item["next_v"],
                        "q": item["q"],
                        "delta_reward": item["delta_reward"],
                        "gamma_delta_v": item["gamma_delta_v"],
                        "delta_q": item["delta_q"],
                    })

        all_support_fractions = [
            value
            for role in role_support.values()
            for value in (role["below_actual_p01_fraction"], role["above_actual_p99_fraction"])
        ]
        if warmup_classification == "DEGRADES":
            counterfactual_reliability = "CONCERN"
        elif all(value == 0.0 for value in all_support_fractions):
            counterfactual_reliability = "SUPPORTED" if warmup_classification == "IMPROVES" else "INCONCLUSIVE"
        else:
            counterfactual_reliability = "INCONCLUSIVE"
        if max_q_error <= Q_ATOL and max_delta_q_error <= Q_ATOL:
            if warmup_classification == "DEGRADES":
                q_integrity_classification = "CANDIDATE_Q_INTEGRITY_CONCERN"
            elif warmup_classification == "IMPROVES" and counterfactual_reliability == "SUPPORTED":
                q_integrity_classification = "CANDIDATE_Q_INTEGRITY_SUPPORTED"
            else:
                q_integrity_classification = "CANDIDATE_Q_INTEGRITY_INCONCLUSIVE"
        else:
            q_integrity_classification = "CANDIDATE_Q_INTEGRITY_CONCERN"

        if warmup_classification in {"NO_CLEAR_IMPROVEMENT", "DEGRADES", "INCONCLUSIVE"}:
            next_stage = "CRITIC_WARMUP_TARGET_AND_VALUE_VALIDITY_GATE"
        elif counterfactual_reliability == "CONCERN":
            next_stage = "COUNTERFACTUAL_CRITIC_RELIABILITY_GATE"
        elif q_integrity_classification == "CANDIDATE_Q_INTEGRITY_SUPPORTED":
            next_stage = "CANDIDATE_SURROGATE_SCREENING_REVALIDATION"
        else:
            next_stage = "CANDIDATE_Q_INTEGRITY_FOLLOWUP"

        formal_seeds_after = derive_episode_seeds(base_seed)
        if tuple(EPISODE_SEED_STREAMS) != tuple(formal_seeds_before.keys()) or formal_seeds_before != formal_seeds_after:
            raise AssertionError("formal episode seed stream definition or values changed")
        if int(cache_q_result["exact_candidate_count"]) != len(candidate_rows):
            raise AssertionError("exact candidate counter differs from the retained component rows")
        if not all(math.isfinite(float(row[key])) for row in candidate_rows for key in (
            "reward", "next_v", "q", "delta_reward", "delta_next_v", "gamma_delta_v",
            "delta_q", "reconstructed_delta_q", "q_reconstruction_error", "delta_q_error",
            "reward_abs_share", "critic_abs_share"
        )):
            raise FloatingPointError("one or more retained Candidate component fields are non-finite")

        result = {
            "provenance": {
                "current_head": current_head,
                "git_dirty_before": "YES" if dirty_before else "NO",
                "git_status_before": git_status_before if git_status_before else "CLEAN",
                "git_status_after_run_before_report_updates": _git("status", "--short"),
                "architecture_version": int(trainer.policy.architecture_version),
                "checkpoint_version": int(trainer._expected_checkpoint_version()),
                "policy_schema": str(trainer.policy.architecture_id),
                "configuration_control_mode": str(diagnostic_context.configuration_control_mode),
                "data_protocol": str(diagnostic_context.data_protocol),
                "train_origin_start": int(diagnostic_context.train_origin_start),
                "train_origin_end": int(diagnostic_context.train_origin_end),
                "terminal_origin": int(diagnostic_context.terminal_origin),
                "episode_slots": int(diagnostic_context.episode_slots),
                "config_sha256": config_digest,
                "formal_episode_seed_stream_names": list(EPISODE_SEED_STREAMS),
                "formal_episode_seed_stream_changed": "NO",
                "warmup_namespace": WARMUP_NAMESPACE,
                "warmup_realization_seed": int(warmup_seed),
                "warmup_generation_rng_seed": int(warmup_generation_seed),
                "warmup_rollout_rng_seed": int(warmup_collection_seed),
                "warmup_critic_optimizer_rng_seed": int(warmup_optimizer_seed),
                "diagnostic_namespace": DIAGNOSTIC_NAMESPACE,
                "diagnostic_realization_seed": int(diagnostic_seed),
                "diagnostic_generation_rng_seed": int(diagnostic_generation_seed),
                "diagnostic_rollout_rng_seed": int(diagnostic_collection_seed),
                "trainer_initialization_rng_seed": int(initialization_seed),
                "critic_optimizer_steps": int(critic_after_q["steps"]),
                "critic_parameter_sha256": str(critic_after_q["digest"]),
                "popart_mean": float(critic_after_q["popart_mean"]),
                "popart_std": float(critic_after_q["popart_std"]),
                "return_stat_count": int(critic_after_q["return_stat_count"]),
                "return_stat_mean": float(critic_after_q["return_stat_mean"]),
                "return_stat_m2": float(critic_after_q["return_stat_m2"]),
                "gamma": float(gamma),
                "gae_lambda": float(gae_lambda),
                "temporary_warmup_checkpoint_sha256": checkpoint_hash,
                "temporary_warmup_checkpoint_metadata": checkpoint_metadata,
                "temporary_realization_and_checkpoint_directory_removed_after_report_validation": "YES",
            },
            "warmup": {
                "real_rollout_executed": "YES",
                "rollout_transition_count": int(len(warmup_rollout.transitions)),
                "critic_steps": int(warmup_metrics["executed_critic_optimizer_steps"]),
                "actor_parameter_delta": float(actor_delta),
                "actor_optimizer_steps": int(trainer.actor_optimizer_steps),
                "critic_parameter_delta_norm": float(warmup_metrics["critic_parameter_delta_norm"]),
                "critic_delta_before_vs_after_nonzero": bool(float(warmup_metrics["critic_parameter_delta_norm"]) > 0.0),
                "cache_teacher_calls": int(warmup_metrics["cache_full_information_teacher_calls"]),
                "uav_exact_teacher_calls": int(warmup_metrics["uav_exact_teacher_calls"]),
                "surrogate_training_count": int(trainer.cache_surrogate_training_count),
                "classification": warmup_classification,
                "popart_raw_value_preservation_error": float(warmup_metrics["popart_raw_value_preservation_error"]),
            },
            "diagnostic_rollout": {
                "executed": "YES",
                "transition_count": int(len(diagnostic_rollout.transitions)),
                "selected_transition_indices": list(diagnostic_indices),
                "selected_origin_indices": diagnostic_origin_indices,
                "selected_position_fractions": [0.10, 0.50, 0.90],
                "reward_to_go_mc": _error_metrics(mc_returns, mc_returns),
                "pre_warmup_critic_vs_mc_return": pre_mc,
                "post_warmup_critic_vs_mc_return": post_mc,
                "pre_warmup_critic_vs_gae_return": pre_gae,
                "post_warmup_critic_vs_gae_return": post_gae,
                "gae_return_vs_mc_return": {
                    "mae": gae_mc["mae"],
                    "rmse": gae_mc["rmse"],
                    "correlation": gae_mc_correlation,
                },
                "actual_state_critic_value_pre_warmup": _percentiles(pre_values),
                "actual_state_critic_value_post_warmup": actual_state_distribution,
                "gamma": float(gamma),
                "gae_lambda": float(gae_lambda),
            },
            "critic_freeze": {
                "before_q_diagnostic": critic_before_q,
                "after_q_diagnostic": critic_after_q,
                "frozen_during_post_warmup_diagnostics": "YES",
                "digest_equal": critic_before_q["digest"] == critic_after_q["digest"],
                "optimizer_steps_equal": critic_before_q["steps"] == critic_after_q["steps"],
                "popart_equal": (
                    critic_before_q["popart_mean"] == critic_after_q["popart_mean"]
                    and critic_before_q["popart_std"] == critic_after_q["popart_std"]
                ),
                "return_statistics_equal": all(
                    critic_before_q[key] == critic_after_q[key]
                    for key in ("return_stat_count", "return_stat_mean", "return_stat_m2")
                ),
            },
            "cache_q": _build_cache_result_section(
                q_components=q_component_rows,
                cache_q_result=cache_q_result,
                candidate_rows=candidate_rows,
                head_summaries=head_summaries,
                max_q_error=max_q_error,
                max_delta_q_error=max_delta_q_error,
                by_role=by_role,
                uav_focus=uav_focus,
                uav_top10=uav_top10,
                actual_state_distribution=actual_state_distribution,
                counterfactual_distribution=counterfactual_distribution,
                role_support=role_support,
            ),
            "classification": {
                "candidate_q_label_integrity": q_integrity_classification,
                "candidate_q_integrity": q_integrity_classification,
                "counterfactual_critic_reliability": counterfactual_reliability,
                "next_stage": next_stage,
                "formal_ppo_update_executed": "NO",
                "actor_optimizer_steps_during_diagnostic": int(trainer.actor_optimizer_steps),
                "one_epoch_training_executed": "NO",
                "ready_for_formal_training_stage": "NO",
            },
        }
        if int(result["cache_q"]["cache_head_count"]) != len(diagnostic_context.action_spec.cache_node_ids):
            raise AssertionError("Cache Q diagnostics do not cover every Cache head")
        if not (
            result["critic_freeze"]["digest_equal"]
            and result["critic_freeze"]["optimizer_steps_equal"]
            and result["critic_freeze"]["popart_equal"]
            and result["critic_freeze"]["return_statistics_equal"]
        ):
            raise AssertionError("post-warmup Critic freeze contract failed")
        result = _assemble_final_result(result)
        _validate_final_result(result)

        # Persist and reload the complete raw result before any Markdown formatting.
        result["provenance"]["formal_result_json_path"] = str(output_json)
        persisted_result = _write_json(output_json, result)
        _validate_final_result(persisted_result)
        for key in (
            "temporary_warmup_checkpoint_sha256",
            "critic_parameter_sha256",
            "warmup_realization_seed",
            "diagnostic_realization_seed",
        ):
            if key not in persisted_result["provenance"]:
                raise AssertionError(f"reloaded final JSON lacks provenance {key}")
        if len(persisted_result["cache_q"]["all_candidate_records"]) != int(persisted_result["cache_q"]["candidate_count"]):
            raise AssertionError("reloaded final JSON candidate rows are incomplete")
        print(f"FINAL_JSON_PERSISTED_AND_RELOADED={output_json}", file=sys.stderr, flush=True)

        second_stage_path = ROOT / "reports" / "second_stage_joint_repair_gate.md"
        one_epoch_path = ROOT / "reports" / "one_epoch_training_gate.md"
        second_stage_before = second_stage_path.read_text(encoding="utf-8")
        one_epoch_before = one_epoch_path.read_text(encoding="utf-8")
        second_stage_after = _apply_rerun_report(second_stage_before, persisted_result)
        one_epoch_after = _apply_one_epoch_report(one_epoch_before, persisted_result)
        second_stage_path.write_text(second_stage_after, encoding="utf-8")
        one_epoch_path.write_text(one_epoch_after, encoding="utf-8")
        second_stage_reloaded = second_stage_path.read_text(encoding="utf-8")
        one_epoch_reloaded = one_epoch_path.read_text(encoding="utf-8")
        _validate_updated_reports(second_stage_reloaded, one_epoch_reloaded)
        if "result-assembly reference error" not in second_stage_reloaded:
            raise AssertionError("first failed attempt provenance was not preserved")
        if "CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION = " + str(persisted_result["CANDIDATE_Q_LABEL_INTEGRITY_CLASSIFICATION"]) not in second_stage_reloaded:
            raise AssertionError("second-stage report classification does not match reloaded result JSON")
        print("MARKDOWN_REPORTS_UPDATED_AND_RELOADED=PASS", file=sys.stderr, flush=True)
        return persisted_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/proposed/system_config.toml",
        help="proposed-joint configuration (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="generation and trainer device")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="import the diagnostic and validate config/schema without generating data or training",
    )
    parser.add_argument(
        "--output-json",
        default="reports/traceable_candidate_q_integrity_result.json",
        help="formal artifact path for every exact Candidate component row",
    )
    parser.add_argument(
        "--assembly-smoke",
        action="store_true",
        help="exercise result assembly, Markdown field access, and JSON I/O with synthetic values only",
    )
    args = parser.parse_args(argv)
    if args.assembly_smoke:
        _synthetic_assembly_smoke()
        print("TRACEABLE_CANDIDATE_Q_RESULT_ASSEMBLY_SMOKE=PASS")
        return 0
    try:
        result = _run_diagnostic(
            args.config,
            device=args.device,
            validate_only=args.validate_only,
            output_json=Path(args.output_json),
        )
    except Exception as exc:
        print(f"TRACEABLE_CANDIDATE_Q_INTEGRITY_GATE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    if args.validate_only:
        print("TRACEABLE_CANDIDATE_Q_INTEGRITY_CONFIG_VALIDATION=PASS")
        return 0
    assert result is not None
    print(f"TRACEABLE_CANDIDATE_Q_INTEGRITY_RESULT={Path(args.output_json).resolve(strict=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
