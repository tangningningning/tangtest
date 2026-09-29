"""根据完整信息计算 Cache 价值，并使用 Exact-oracle Hybrid 筛选候选。

KEEP 始终精确评估。只有滚动 Exact 质量门通过后，策略无关的 delta-Q
集成模型才筛选合法 UPDATE 候选；最终选中项由正式精确分支执行器复核。
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from decision import cache_configuration, slot_environment, uav_configuration

from .primitive_control import (
    _all_cache_target_configurations_with_features,
    _intermediate_observation,
    _target_cache_proposals_with_features,
)
from .rollout import _check_request_contract, _check_reward


OPERATION_ORDER = ("NO_CHANGE", "ADD", "DELETE", "SWAP")
CACHE_SURROGATE_K_CANDIDATES = (4, 6, 8, 10, 12, 16)


class CacheCounterfactualDeltaQSurrogate(nn.Module):
    """根据 Cache 上下文与候选特征预测相对 KEEP 的 delta-Q。网络不依赖 PPO 策略或其编码器，可使用独立优化器。"""

    architecture_id = "cache_counterfactual_delta_q_surrogate_mlp"
    architecture_version = 1

    def __init__(
        self,
        context_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if int(context_dim) <= 0 or int(candidate_feature_dim) <= 0:
            raise ValueError("surrogate input dimensions must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("surrogate hidden_dim must be positive")
        self.context_dim = int(context_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.context_dim + self.candidate_feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, conditional_context: Any, candidate_feature: Any) -> torch.Tensor:
        context = torch.as_tensor(
            conditional_context,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        features = torch.as_tensor(
            candidate_feature,
            dtype=torch.float32,
            device=context.device,
        )
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if context.ndim != 2 or features.ndim != 3:
            raise ValueError(
                "surrogate inputs must have shapes [B,D] and [B,C,F]"
            )
        if int(context.shape[1]) != self.context_dim:
            raise ValueError(
                f"conditional_context width={int(context.shape[1])}, "
                f"expected={self.context_dim}"
            )
        if int(features.shape[0]) != int(context.shape[0]):
            raise ValueError("surrogate context/candidate batch dimensions differ")
        if int(features.shape[2]) != self.candidate_feature_dim:
            raise ValueError(
                f"candidate_feature width={int(features.shape[2])}, "
                f"expected={self.candidate_feature_dim}"
            )
        if not bool(torch.all(torch.isfinite(context)).item()):
            raise ValueError("conditional_context contains NaN/Inf")
        if not bool(torch.all(torch.isfinite(features)).item()):
            raise ValueError("candidate_feature contains NaN/Inf")
        expanded_context = context.unsqueeze(1).expand(
            -1, int(features.shape[1]), -1
        )
        values = self.network(
            torch.cat([expanded_context, features], dim=-1)
        ).squeeze(-1)
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise FloatingPointError("surrogate delta-Q output contains NaN/Inf")
        return values


def _position_dict(values: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value, dtype=np.float64).copy()
        for key, value in values.items()
    }


def _production_action_map(transition: Any) -> dict[int, dict[str, Any]]:
    actions = tuple(dict(value) for value in transition.cache_conditional_actions)
    mapping = {int(item["head_index"]): item for item in actions}
    expected = set(range(len(transition.previous_cache_state)))
    if set(mapping) != expected:
        raise RuntimeError(
            "proposed production Cache action record is not one entry per node: "
            f"got={sorted(mapping)}, expected={sorted(expected)}"
        )
    return mapping


def _replay_cache_prefix(transition: Any, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    previous_cache = np.asarray(transition.previous_cache_state, dtype=np.int8).copy()
    production_final = np.asarray(transition.next_cache_state, dtype=np.int8)
    working = previous_cache.copy()
    action_map = _production_action_map(transition)
    for index in range(int(target_index)):
        working[index] = production_final[index]
        is_keep = str(action_map[index].get("conditional_action", "")).upper() == "KEEP" or (
            str(action_map[index].get("operation_type", "")).upper() == "NO_CHANGE"
        )
        if is_keep and not np.array_equal(
            working[index], previous_cache[index]
        ):
            raise RuntimeError("production KEEP changed a Cache row")
    return working, previous_cache


def _stored_node_cache_trigger_aux(
    policy: Any,
    transition: Any,
    node_index: int,
    node_count: int,
) -> np.ndarray:
    """读取 rollout 中冻结的 Cache Trigger 辅助状态。"""

    auxiliary_dim = getattr(policy, "trigger_aux_dim", None)
    if auxiliary_dim is None:
        raise RuntimeError("正式 Cache Trigger 策略未提供 trigger_aux_dim。")
    stored_auxiliary = getattr(transition, "cache_trigger_aux", None)
    if stored_auxiliary is None:
        raise RuntimeError(
            "Cache Trigger 信用计算缺少 rollout 保存的 cache_trigger_aux。"
        )
    auxiliary = np.asarray(stored_auxiliary, dtype=np.float32)
    expected_shape = (int(node_count), int(auxiliary_dim))
    if tuple(auxiliary.shape) != expected_shape:
        raise RuntimeError(
            "Cache Trigger 辅助状态形状为 "
            f"{tuple(auxiliary.shape)}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(auxiliary)):
        raise FloatingPointError("保存的 Cache Trigger 辅助状态不是有限值。")
    return auxiliary[int(node_index)].copy()


def _masked_operation_probabilities(
    policy: Any,
    conditional_context: np.ndarray,
    node_index: int,
    operation_mask: np.ndarray,
    trigger_aux: np.ndarray | None = None,
) -> np.ndarray:
    if trigger_aux is None:
        raise RuntimeError("Cache Trigger logits 缺少保存的辅助状态。")
    logits, single = policy._operation_logits(conditional_context, trigger_aux)
    if not single:
        raise RuntimeError("Cache operation distribution requires one context")
    raw_logits = logits[int(node_index)]
    mask = torch.as_tensor(operation_mask, dtype=torch.bool, device=raw_logits.device)
    distribution = torch.distributions.Categorical(
        logits=raw_logits.masked_fill(~mask, -float("inf"))
    )
    probabilities = distribution.probs.detach().cpu().numpy().astype(np.float64)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Cache operation probabilities are invalid")
    probabilities /= total
    return probabilities


def _candidate_probabilities(
    policy: Any,
    conditional_context: np.ndarray,
    candidate_features: np.ndarray,
) -> np.ndarray:
    scores, single = policy._candidate_scores(
        conditional_context,
        candidate_features,
        policy.cache_feature_dim,
        policy.cache_candidate_scorer,
    )
    if not single:
        raise RuntimeError("Cache candidate distribution requires one context")
    probabilities = torch.softmax(scores.squeeze(0), dim=-1)
    return probabilities.detach().cpu().numpy().astype(np.float64)


def _node_policy_state(
    trainer: Any,
    transition: Any,
    step: Any,
    node_index: int,
    feature_dim: int,
) -> dict[str, Any]:
    context = trainer.context
    trigger_aux = _stored_node_cache_trigger_aux(
        trainer.policy,
        transition,
        node_index,
        len(context.action_spec.cache_node_ids),
    )
    working_cache, previous_cache = _replay_cache_prefix(transition, node_index)
    current_uav = _position_dict(transition.previous_uav_positions)
    proposal_cache_nodes = uav_configuration.cache_nodes_with_uav_positions(
        context.cache_nodes,
        current_uav,
    )
    conditional_context = _intermediate_observation(
        working_cache,
        current_uav,
        step,
        context,
    )
    # 正式主线枚举完整目标配置。候选张量不包含操作标签，proposal_source 只记录来源。
    action_map = _production_action_map(transition)
    sequential_triggers = np.zeros(
        len(context.action_spec.cache_node_ids), dtype=np.int8
    )
    for index in range(node_index):
        if str(action_map[index].get("conditional_action", "")).upper() != "KEEP":
            sequential_triggers[index] = 1
    sequential_triggers[node_index] = 1
    update_candidates, unified_features = _target_cache_proposals_with_features(
        working_cache,
        previous_cache,
        sequential_triggers,
        node_index,
        np.asarray(conditional_context, dtype=np.float32),
        step,
        context,
        proposal_cache_nodes,
        surrogate_predictor=(
            trainer._predict_cache_surrogate
            if getattr(trainer, "cache_surrogate_ensemble", None) is not None
            else None
        ),
        surrogate_ready=bool(
            getattr(trainer, "cache_surrogate_screening_ready", False)
            and getattr(trainer, "cache_surrogate_screening_active", False)
        ),
    )
    if update_candidates:
        candidate_probabilities = _candidate_probabilities(
            trainer.policy,
            np.asarray(conditional_context, dtype=np.float32),
            unified_features,
        )
        candidate_features = [
            np.asarray(feature, dtype=np.float32)
            for feature in unified_features
        ]
        candidate_operations = np.asarray(
            np.zeros(len(update_candidates), dtype=np.int64),
            dtype=np.int64,
        )
    else:
        candidate_probabilities = np.zeros(0, dtype=np.float64)
        candidate_features = []
        candidate_operations = np.zeros(0, dtype=np.int64)
    operation_mask = np.asarray([True, bool(update_candidates)], dtype=bool)
    operation_probabilities = _masked_operation_probabilities(
        trainer.policy,
        np.asarray(conditional_context, dtype=np.float32),
        node_index,
        operation_mask,
        trigger_aux=trigger_aux,
    )
    p_keep = float(operation_probabilities[0])
    p_update = float(operation_probabilities[1])
    if candidate_probabilities.size:
        candidate_probabilities = candidate_probabilities / float(
            candidate_probabilities.sum()
        )
    return {
        "working_cache": working_cache,
        "previous_cache": previous_cache,
        "conditional_context": np.asarray(conditional_context, dtype=np.float32),
        "operation_mask": operation_mask,
        "operation_probabilities": operation_probabilities,
        "p_keep": p_keep,
        "p_update": p_update,
        "candidates": update_candidates,
        "candidate_features": candidate_features,
        "candidate_operations": candidate_operations,
        "candidate_probabilities": np.asarray(candidate_probabilities, dtype=np.float64),
        "grouped_candidates": {
            name: [] for name in OPERATION_ORDER[1:]
        },
        "grouped_features": {
            name: np.zeros((0, int(feature_dim)), dtype=np.float32)
            for name in OPERATION_ORDER[1:]
        },
    }


def _execute_cache_branch(
    trainer: Any,
    transition: Any,
    step: Any,
    next_step: Any | None,
    node_index: int,
    state: dict[str, Any],
    candidate: Any | None,
) -> dict[str, Any]:
    context = trainer.context
    previous_cache = np.asarray(transition.previous_cache_state, dtype=np.int8).copy()
    previous_uav = _position_dict(transition.previous_uav_positions)
    final_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
    final_uav = _position_dict(transition.next_uav_positions)
    production_triggers, production_uav_triggers = context.action_spec.action_vector_to_triggers(
        transition.action
    )
    branch_triggers = np.asarray(production_triggers, dtype=np.int8).copy()
    production_actions = _production_action_map(transition)
    branch_actions: list[dict[str, Any]] = []
    for index, node_id in enumerate(context.action_spec.cache_node_ids):
        action = dict(production_actions[index])
        if index == node_index:
            if candidate is None:
                branch_triggers[index] = 0
                final_cache[index] = state["working_cache"][index]
                action = {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": int(index),
                    "candidate_count": 0,
                    "selected_index": None,
                    "conditional_action": "KEEP",
                    "actual_changed": False,
                }
            else:
                branch_triggers[index] = 1
                final_cache[index] = np.asarray(
                    candidate.target_node_cache, dtype=np.int8
                ).copy()
                action = {
                    "head_type": "cache",
                    "head_id": str(node_id),
                    "head_index": int(index),
                    "candidate_count": -1,
                    "selected_index": None,
                    "conditional_action": "TARGET_CONFIGURATION",
                    "target_cache": np.asarray(
                        candidate.target_node_cache, dtype=np.int8
                    ).astype(int).tolist(),
                    "proposal_source": str(candidate.proposal_source),
                    "actual_changed": bool(
                        not np.array_equal(
                            candidate.before_node_cache,
                            candidate.target_node_cache,
                        )
                    ),
                }
        branch_actions.append(action)
    branch_actions.sort(key=lambda item: int(item["head_index"]))
    production_uav_triggers = {
        str(key): int(value) for key, value in production_uav_triggers.items()
    }
    configuration_override = slot_environment.build_policy_target_configuration(
        previous_cache,
        previous_uav,
        final_cache,
        final_uav,
        branch_triggers,
        production_uav_triggers,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        cache_actions=tuple(branch_actions),
        uav_actions=tuple(dict(value) for value in transition.uav_conditional_actions),
    )
    result = slot_environment.execute_slot(
        previous_cache,
        previous_uav,
        branch_triggers,
        production_uav_triggers,
        step.scenario_slice,
        step.actual_tasks,
        step.actual_vehicle_positions,
        context.local_cache,
        context.service_catalog,
        context.cfg,
        context.cache_nodes,
        context.cache_cfg,
        context.flight_boundary,
        slot_id=step.slot,
        time_s=step.time_s,
        performance_profile=None,
        configuration_override=configuration_override,
    )
    if not result.step_feasible:
        raise RuntimeError(
            "Exact Cache counterfactual is infeasible: "
            f"origin={step.origin_index}, node={node_index}"
        )
    _check_request_contract(previous_cache, previous_uav, result, context.action_spec)
    reward = _check_reward(result)
    next_observation = None
    if next_step is not None:
        next_observation = _intermediate_observation(
            np.asarray(result.next_cache_state, dtype=np.int8),
            result.next_uav_positions,
            next_step,
            context,
        )
    return {
        "reward": float(reward),
        "next_observation": next_observation,
        "result": result,
    }


def _cache_branch_identity(
    transition: Any,
    node_index: int,
    state: dict[str, Any],
    candidate: Any | None,
) -> tuple[int, bytes, tuple[tuple[str, float, float], ...]]:
    """返回一个 Cache 分支的完整物理状态标识。transition origin 用于标识固定任务、车辆与预测输入；其余部分包含完整的配置后 Cache 状态和全部 UAV 位置。不使用 Operation 标签或候选索引；只有完整物理状态相同的逻辑候选才可复用。"""

    final_cache = np.asarray(transition.next_cache_state, dtype=np.int8).copy()
    if candidate is None:
        final_cache[int(node_index)] = np.asarray(
            state["working_cache"][int(node_index)], dtype=np.int8
        )
    else:
        final_cache[int(node_index)] = np.asarray(
            candidate.target_node_cache, dtype=np.int8
        )
    final_uav = tuple(
        sorted(
            (
                str(node_id),
                float(np.asarray(xy, dtype=np.float64)[0]),
                float(np.asarray(xy, dtype=np.float64)[1]),
            )
            for node_id, xy in transition.next_uav_positions.items()
        )
    )
    return int(transition.origin_index), final_cache.tobytes(), final_uav


def _branch_q_components(
    trainer: Any, branches: list[dict[str, Any]]
) -> dict[str, Any]:
    """返回 Exact-Q 分支奖励、下一状态 Critic 价值及各项 Q 组成。"""

    observations = [branch["next_observation"] for branch in branches]
    if any(value is None for value in observations):
        if any(value is not None for value in observations):
            raise RuntimeError("Cache branch next observations are partially missing")
        next_values = np.zeros(len(branches), dtype=np.float64)
    else:
        observation_batch = np.stack(
            [np.asarray(value, dtype=np.float32) for value in observations], axis=0
        )
        with torch.no_grad():
            value_tensor = trainer.policy.critic_value(observation_batch)
        next_values = value_tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    rewards = np.asarray([branch["reward"] for branch in branches], dtype=np.float64)
    q_values = rewards + float(trainer.hyperparameters.gamma) * next_values
    if not (
        np.all(np.isfinite(rewards))
        and np.all(np.isfinite(next_values))
        and np.all(np.isfinite(q_values))
    ):
        raise FloatingPointError("Exact Cache Q components are non-finite")
    reconstruction_error = np.abs(
        q_values
        - (rewards + float(trainer.hyperparameters.gamma) * next_values)
    )
    if not np.all(np.isfinite(reconstruction_error)):
        raise FloatingPointError("Exact Cache Q reconstruction error is non-finite")
    return {
        "branch_reward": rewards,
        "branch_next_critic_value": next_values,
        "branch_q": q_values,
        "max_reconstruction_error": float(
            np.max(reconstruction_error) if len(reconstruction_error) else 0.0
        ),
    }


def _branch_q_values(trainer: Any, branches: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        _branch_q_components(trainer, branches)["branch_q"], dtype=np.float64
    )


def _rank_correlation(predicted: np.ndarray, exact: np.ndarray) -> float:
    """计算不依赖第三方库的 Spearman 风格秩相关系数。"""

    predicted_a = np.asarray(predicted, dtype=np.float64).reshape(-1)
    exact_a = np.asarray(exact, dtype=np.float64).reshape(-1)
    if len(predicted_a) < 2 or len(exact_a) < 2:
        return float("nan")
    if not np.all(np.isfinite(predicted_a)) or not np.all(np.isfinite(exact_a)):
        raise FloatingPointError("Cache surrogate rank inputs are non-finite")

    def _average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
            start = stop
        return ranks

    predicted_ranks = _average_ranks(predicted_a)
    exact_ranks = _average_ranks(exact_a)
    predicted_centered = predicted_ranks - float(np.mean(predicted_ranks))
    exact_centered = exact_ranks - float(np.mean(exact_ranks))
    denominator = float(
        np.sqrt(np.sum(predicted_centered**2) * np.sum(exact_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(predicted_centered * exact_centered) / denominator)


def _hybrid_exact_indices(
    predicted_delta: np.ndarray,
    predicted_sigma: np.ndarray,
    selected_k: int,
    *,
    anchor_indices: tuple[int, ...] = (),
    production_selected_index: int | None = None,
    random_audit_index: int | None = None,
) -> list[int]:
    """从完整可行候选集合中选择 Exact-Q 子集。"""

    predicted = np.asarray(predicted_delta, dtype=np.float64).reshape(-1)
    sigma = np.asarray(predicted_sigma, dtype=np.float64).reshape(-1)
    if predicted.shape != sigma.shape or not np.all(np.isfinite(predicted)) or not np.all(
        np.isfinite(sigma)
    ):
        raise ValueError("full-universe Hybrid scores are invalid")
    if not len(predicted):
        return []
    k = max(1, min(int(selected_k), len(predicted)))
    selected = set(int(index) for index in anchor_indices)
    if production_selected_index is not None:
        selected.add(int(production_selected_index))
    selected.update(int(index) for index in np.argsort(-predicted, kind="mergesort")[:k])
    ucb = predicted + 2.0 * np.maximum(sigma, 0.0)
    selected.update(int(index) for index in np.argsort(-ucb, kind="mergesort")[:k])
    selected.add(int(np.argmax(sigma)))
    if random_audit_index is not None:
        selected.add(int(random_audit_index))
    if any(index < 0 or index >= len(predicted) for index in selected):
        raise IndexError("Hybrid Exact-Q subset index is out of range")
    return sorted(selected)


def _screening_gate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 full-Exact 行，用于选择 K 和执行安全 gate。"""

    recall_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    regret_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    normalized_regret_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    top3_overlap_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    trigger_sign_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    rank_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    mae_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    squared_residual_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    exact_count_values: dict[int, list[float]] = {
        k: [] for k in CACHE_SURROGATE_K_CANDIDATES
    }
    for row in rows:
        predicted = np.asarray(row["predicted_delta_q"], dtype=np.float64).reshape(-1)
        exact_q = np.asarray(row["exact_q"], dtype=np.float64).reshape(-1)
        if len(predicted) != len(exact_q) or not len(predicted):
            continue
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(exact_q)):
            raise FloatingPointError("Cache screening gate inputs are non-finite")
        best_q = float(np.max(exact_q))
        best_mask = np.isclose(exact_q, best_q, rtol=1.0e-10, atol=1.0e-10)
        q_range = float(np.max(exact_q) - np.min(exact_q))
        exact_top3 = set(np.argsort(-exact_q, kind="mergesort")[: min(3, len(exact_q))])
        anchors = tuple(int(value) for value in row.get("anchor_indices", ()))
        production_index_value = row.get("production_selected_index")
        production_index = (
            None if production_index_value is None else int(production_index_value)
        )
        random_index_value = row.get("random_audit_index")
        random_index = None if random_index_value is None else int(random_index_value)
        for k in CACHE_SURROGATE_K_CANDIDATES:
            exact_indices = _hybrid_exact_indices(
                predicted,
                np.asarray(row["predicted_sigma"], dtype=np.float64),
                k,
                anchor_indices=anchors,
                production_selected_index=production_index,
                random_audit_index=random_index,
            )
            hybrid = predicted.copy()
            hybrid[np.asarray(exact_indices, dtype=np.int64)] = exact_q[
                np.asarray(exact_indices, dtype=np.int64)
            ]
            ranking = np.argsort(-hybrid, kind="mergesort")
            hybrid_best = int(ranking[0])
            recall_values[k].append(float(best_mask[hybrid_best]))
            regret = max(0.0, best_q - float(exact_q[hybrid_best]))
            normalized = regret / (q_range + 1.0e-12)
            regret_values[k].append(regret)
            normalized_regret_values[k].append(normalized)
            hybrid_top3 = set(ranking[: min(3, len(ranking))])
            top3_overlap_values[k].append(
                float(len(exact_top3 & hybrid_top3) / max(len(exact_top3), 1))
            )
            trigger_sign_values[k].append(
                float((best_q >= 0.0) == (float(np.max(hybrid)) >= 0.0))
            )
            rank = _rank_correlation(hybrid, exact_q)
            if np.isfinite(rank):
                rank_values[k].append(rank)
            residual = hybrid - exact_q
            mae_values[k].extend(np.abs(residual).tolist())
            squared_residual_values[k].extend(np.square(residual).tolist())
            exact_count_values[k].append(float(len(exact_indices)))

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90.0)),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(np.max(array)),
        }

    recall_at_k = {
        str(k): float(np.mean(values))
        for k, values in recall_values.items()
        if values
    }
    regret_by_k = {
        str(k): _stats(values)
        for k, values in regret_values.items()
        if values
    }
    normalized_regret_by_k = {
        str(k): _stats(values)
        for k, values in normalized_regret_values.items()
        if values
    }
    normalized_p95 = {
        str(k): float(stats["p95"])
        for k, stats in normalized_regret_by_k.items()
        if stats["p95"] is not None
    }
    return {
        "row_count": int(len(rows)),
        "recall_at_k": recall_at_k,
        "best_miss_rate_at_k": {
            key: float(1.0 - value) for key, value in recall_at_k.items()
        },
        "regret_at_k": regret_by_k,
        "normalized_regret_at_k": normalized_regret_by_k,
        "normalized_regret_p95_at_k": normalized_p95,
        "top3_overlap_at_k": {
            str(k): float(np.mean(values)) for k, values in top3_overlap_values.items() if values
        },
        "trigger_sign_agreement_at_k": {
            str(k): float(np.mean(values)) for k, values in trigger_sign_values.items() if values
        },
        "rank_correlation_at_k": {
            str(k): float(np.mean(values)) for k, values in rank_values.items() if values
        },
        "delta_q_mae_at_k": {
            str(k): float(np.mean(values)) for k, values in mae_values.items() if values
        },
        "delta_q_rmse_at_k": {
            str(k): float(np.sqrt(np.mean(values)))
            for k, values in squared_residual_values.items()
            if values
        },
        "exact_evaluated_count_mean_at_k": {
            str(k): float(np.mean(values)) for k, values in exact_count_values.items() if values
        },
        "rank_correlation": (
            None
            if not rank_values[CACHE_SURROGATE_K_CANDIDATES[-1]]
            else float(np.mean(rank_values[CACHE_SURROGATE_K_CANDIDATES[-1]]))
        ),
    }


def _selected_cache_candidate_index(
    transition: Any,
    node_index: int,
    candidates: list[Any],
) -> int | None:
    """解析正式 target 候选，不将 padding 当作有效数据。"""

    if not candidates:
        return None
    operation_actions = getattr(transition, "cache_operation_actions", None)
    if operation_actions is not None and int(operation_actions[node_index]) == 0:
        return None
    actions = _production_action_map(transition)
    action = actions.get(int(node_index), {})
    target = action.get("target_cache")
    target_array = None if target is None else np.asarray(target, dtype=np.int8)
    selected_indices = getattr(transition, "cache_selected_indices", None)
    if selected_indices is not None:
        selected_index = int(selected_indices[node_index])
        if 0 <= selected_index < len(candidates) and (
            target_array is None
            or np.array_equal(
                target_array,
                np.asarray(candidates[selected_index].target_node_cache, dtype=np.int8),
            )
        ):
            return selected_index
    if target_array is None:
        return None
    for candidate_index, candidate in enumerate(candidates):
        if np.array_equal(
            target_array,
            np.asarray(candidate.target_node_cache, dtype=np.int8),
        ):
            return int(candidate_index)
    return None


def _stable_masked_softmax(
    values: np.ndarray,
    mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    q_values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    scale = float(temperature)
    if q_values.shape != valid.shape or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cache teacher Q/mask/temperature is invalid")
    probabilities = np.zeros_like(q_values, dtype=np.float64)
    if np.any(valid):
        legal_logits = q_values[valid] / scale
        if not np.all(np.isfinite(legal_logits)):
            raise FloatingPointError("Cache teacher received non-finite legal Q values")
        legal_logits -= float(np.max(legal_logits))
        weights = np.exp(legal_logits)
        probabilities[valid] = weights / float(np.sum(weights))
    return probabilities


def _robust_mad_scale(values: np.ndarray, epsilon: float, maximum: float) -> float:
    """根据训练数据估计有界尺度，降低异常值的影响。"""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    lower = float(epsilon)
    upper = float(maximum)
    if not len(array) or not math.isfinite(lower) or lower <= 0.0:
        return lower
    if not math.isfinite(upper) or upper < lower:
        raise ValueError("Cache teacher scale bounds are invalid")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    robust_std = 1.4826 * mad
    typical_magnitude = float(np.median(np.abs(array)))
    estimate = max(robust_std, typical_magnitude, lower)
    return float(np.clip(estimate, lower, upper))


def state_head_candidate_teacher(
    values: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """构造一个局部归一化的 Candidate teacher 及其诊断指标。"""

    q_values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    lower = float(epsilon)
    if q_values.shape != valid.shape or not math.isfinite(lower) or lower <= 0.0:
        raise ValueError("Candidate teacher Q/mask/epsilon is invalid")
    probabilities = np.zeros_like(q_values, dtype=np.float64)
    legal_q = q_values[valid]
    if not len(legal_q):
        return probabilities, {
            "candidate_count": 0,
            "q_min": None,
            "q_max": None,
            "q_range": None,
            "median": None,
            "mad": None,
            "resolved_scale": None,
            "fallback_type": "NO_LEGAL_CANDIDATE",
            "teacher_entropy": None,
            "teacher_top1_probability": None,
            "teacher_top1_top2_probability_gap": None,
            "teacher_argmax": None,
            "normalized_top1_top2_q_gap": None,
        }
    if not np.all(np.isfinite(legal_q)):
        raise FloatingPointError("Candidate teacher received non-finite legal Q values")

    center = float(np.median(legal_q))
    absolute_deviation = np.abs(legal_q - center)
    mad = float(np.median(absolute_deviation))
    robust_scale = 1.4826 * mad
    standard_deviation = float(np.std(legal_q))
    value_range = float(np.max(legal_q) - np.min(legal_q))
    equivalent = value_range <= lower
    if equivalent:
        scale = lower
        fallback_type = "EQUAL_Q_EPSILON"
        normalized = np.zeros_like(legal_q)
    else:
        scale_candidates = (
            (robust_scale, "MAD_1P4826"),
            (standard_deviation, "STD_FALLBACK"),
            (value_range, "RANGE_FALLBACK"),
            (lower, "EPSILON_FALLBACK"),
        )
        scale, fallback_type = next(
            (float(candidate), label)
            for candidate, label in scale_candidates
            if math.isfinite(float(candidate)) and float(candidate) > 0.0
        )
        normalized = (legal_q - center) / scale
    if not math.isfinite(scale) or scale <= 0.0 or not np.all(np.isfinite(normalized)):
        raise FloatingPointError("Candidate state-head normalization is non-finite")

    legal_probabilities = _stable_masked_softmax(
        normalized,
        np.ones(len(normalized), dtype=bool),
        1.0,
    )
    probabilities[valid] = legal_probabilities
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("Candidate state-head teacher is non-finite")
    ordered_probabilities = np.sort(legal_probabilities)[::-1]
    ordered_q = np.sort(legal_q)[::-1]
    entropy = float(
        -np.sum(
            legal_probabilities
            * np.log(np.maximum(legal_probabilities, 1.0e-300))
        )
    )
    top_gap = (
        float(ordered_probabilities[0] - ordered_probabilities[1])
        if len(ordered_probabilities) > 1
        else 1.0
    )
    normalized_q_gap = (
        float((ordered_q[0] - ordered_q[1]) / scale)
        if len(ordered_q) > 1
        else 0.0
    )
    legal_indices = np.flatnonzero(valid)
    teacher_argmax = int(legal_indices[int(np.argmax(legal_probabilities))])
    return probabilities, {
        "candidate_count": int(len(legal_q)),
        "q_min": float(np.min(legal_q)),
        "q_max": float(np.max(legal_q)),
        "q_range": value_range,
        "median": center,
        "mad": mad,
        "resolved_scale": float(scale),
        "fallback_type": fallback_type,
        "teacher_entropy": entropy,
        "teacher_top1_probability": float(ordered_probabilities[0]),
        "teacher_top1_top2_probability_gap": top_gap,
        "teacher_argmax": teacher_argmax,
        "normalized_top1_top2_q_gap": normalized_q_gap,
    }


def _cache_teacher_probability_stats(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    *,
    trigger: bool,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if trigger:
        if probs.shape[:-1] != valid.shape or probs.shape[-1] != 2:
            raise ValueError("Cache Trigger teacher statistics have invalid shapes")
        update = probs[..., 1][valid]
        entropy = -np.sum(
            np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1.0e-300)), 0.0),
            axis=-1,
        )[valid]
        return {
            "entropy_mean": float(np.mean(entropy)) if len(entropy) else 0.0,
            "entropy_median": float(np.median(entropy)) if len(entropy) else 0.0,
            "p_update_lt_0_01": float(np.mean(update < 0.01)) if len(update) else 0.0,
            "p_update_gt_0_99": float(np.mean(update > 0.99)) if len(update) else 0.0,
            "p_update_lt_0_05": float(np.mean(update < 0.05)) if len(update) else 0.0,
            "p_update_gt_0_95": float(np.mean(update > 0.95)) if len(update) else 0.0,
        }
    if probs.shape != valid.shape or probs.ndim < 2:
        raise ValueError("Cache Candidate teacher statistics have invalid shapes")
    entropy_values: list[float] = []
    top1_values: list[float] = []
    top3_values: list[float] = []
    near_one_hot: list[float] = []
    for row, mask in zip(probs.reshape(-1, probs.shape[-1]), valid.reshape(-1, valid.shape[-1])):
        legal = row[mask]
        if not len(legal):
            continue
        ordered = np.sort(legal)[::-1]
        entropy_values.append(float(-np.sum(legal * np.log(np.maximum(legal, 1.0e-300)))) )
        top1_values.append(float(ordered[0]))
        top3_values.append(float(np.sum(ordered[: min(3, len(ordered))])))
        near_one_hot.append(float(ordered[0] >= 0.99))
    return {
        "entropy_mean": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "top1_probability_mean": float(np.mean(top1_values)) if top1_values else 0.0,
        "top3_probability_mass_mean": float(np.mean(top3_values)) if top3_values else 0.0,
        "near_one_hot_fraction": float(np.mean(near_one_hot)) if near_one_hot else 0.0,
    }


def evaluate_cache_full_information_values(
    trainer: Any,
    rollout: Any,
    *,
    target_transition_indices: tuple[int, ...] | None = None,
    target_node_indices: tuple[int, ...] | None = None,
    trigger_temperature: float | None = None,
    candidate_temperature: float | None = None,
    force_full_exact: bool = False,
    train_surrogate: bool = True,
    include_q_component_diagnostics: bool = False,
    include_full_candidate_q_components: bool = False,
) -> dict[str, Any]:
    """计算固定后续决策下的 Cache 价值和 detached full-information teacher。FULL_EXACT 是 oracle。Surrogate 通过滚动 exact-audit gate 后，同一 evaluator 使用 HYBRID_EXACT_SURROGATE 补全候选。KEEP 始终精确评估，选中的精确分支会覆盖其预测值。"""

    trigger_scale = None if trigger_temperature is None else float(trigger_temperature)
    readiness_check = getattr(trainer, "_require_critic_bootstrap_teacher_ready", None)
    if readiness_check is not None:
        readiness_check("Cache full-information teacher/surrogate evaluation")
    if include_full_candidate_q_components and not force_full_exact:
        raise ValueError(
            "full candidate Q components require explicit force_full_exact=True"
        )
    candidate_scale = 1.0
    if trigger_scale is not None and (
        not math.isfinite(trigger_scale) or trigger_scale <= 0.0
    ):
        raise ValueError("Cache Trigger teacher temperature must be positive and finite")
    if candidate_temperature is not None and not math.isclose(
        float(candidate_temperature), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "state-head Candidate teacher temperature must equal 1.0"
        )
    transitions = tuple(rollout.transitions)
    if not transitions:
        raise ValueError("full-information Cache evaluation requires transitions")
    if target_transition_indices is None:
        selected_indices = tuple(range(len(transitions)))
    else:
        selected_indices = tuple(int(index) for index in target_transition_indices)
        if not selected_indices or len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Cache evaluator transition indices must be non-empty and unique")
        if any(index < 0 or index >= len(transitions) for index in selected_indices):
            raise IndexError("Cache evaluator transition index is out of range")
    node_count = len(trainer.context.action_spec.cache_node_ids)
    if target_node_indices is None:
        node_indices = tuple(range(node_count))
    else:
        node_indices = tuple(int(index) for index in target_node_indices)
        if not node_indices or len(set(node_indices)) != len(node_indices):
            raise ValueError("Cache evaluator node indices must be non-empty and unique")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise IndexError("Cache evaluator node index is out of range")
    if not bool(getattr(trainer.policy, "true_binary_trigger", False)):
        raise RuntimeError("full-information Cache teacher requires binary KEEP/UPDATE")
    if rollout.cache_candidate_features is None:
        raise ValueError("rollout does not expose Cache candidate tensor capacity")
    candidate_capacity = int(rollout.cache_candidate_features.shape[2])
    feature_dim = int(rollout.cache_candidate_features.shape[3])

    selected_transitions = tuple(transitions[index] for index in selected_indices)
    origin_start = min(int(item.origin_index) for item in selected_transitions)
    origin_end = max(int(item.origin_index) for item in selected_transitions)
    support_count = origin_end - origin_start + 1
    if not any(
        bool(item.done) and int(item.origin_index) == origin_end
        for item in selected_transitions
    ):
        support_count += 1
    from .interface import aligned_train_steps

    aligned_steps = aligned_train_steps(
        trainer.context,
        origin_start,
        support_count,
        include_bootstrap_step=False,
    )
    steps_by_origin = {int(item.origin_index): item for item in aligned_steps}
    sample_count = len(selected_transitions)
    q_keep = np.full((sample_count, node_count), np.nan, dtype=np.float64)
    candidate_q = np.full(
        (sample_count, node_count, candidate_capacity), np.nan, dtype=np.float64
    )
    candidate_valid = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=bool
    )
    candidate_features = np.zeros(
        (sample_count, node_count, candidate_capacity, feature_dim), dtype=np.float32
    )
    candidate_operation_indices = np.zeros(
        (sample_count, node_count, candidate_capacity), dtype=np.int64
    )
    candidate_policy_probabilities = np.zeros_like(candidate_q)
    candidate_predicted_delta_q = np.zeros_like(candidate_q)
    candidate_predicted_sigma = np.zeros_like(candidate_q)
    candidate_exact_evaluated = np.zeros_like(candidate_valid)
    candidate_q_sources = np.full(candidate_q.shape, "", dtype=object)
    trigger_policy_probabilities = np.full(
        (sample_count, node_count, 2), np.nan, dtype=np.float64
    )
    candidate_counts = np.zeros((sample_count, node_count), dtype=np.int64)
    environment_executions = 0
    deduplicated_branches = 0
    logical_candidate_count = 0
    keep_branch_count = 0
    exact_candidate_count = 0
    surrogate_candidate_count = 0
    confirmation_added_exact_count = 0
    full_exact_audit_count = 0
    full_exact_environment_baseline = 0
    surrogate_inference_runtime_seconds = 0.0
    surrogate_training_metrics: dict[str, Any] = {
        "count": 0,
        "loss": math.nan,
        "mae": math.nan,
        "rmse": math.nan,
    }
    exact_context_labels: list[np.ndarray] = []
    exact_feature_labels: list[np.ndarray] = []
    exact_delta_labels: list[float] = []
    exact_group_labels: list[int] = []
    screening_rows: list[dict[str, Any]] = []
    q_component_diagnostics: list[list[dict[str, Any]]] = [
        [] for _ in range(sample_count)
    ]
    full_candidate_q_components: list[dict[str, Any]] = []
    max_q_component_reconstruction_error = 0.0
    feasible_universe_counts: list[int] = []
    policy_best_recalled: list[float] = []
    surrogate_ready = bool(
        not force_full_exact
        and getattr(trainer, "cache_surrogate_screening_ready", False)
        and getattr(trainer, "cache_surrogate_screening_active", False)
    )
    surrogate_rng = getattr(trainer, "cache_surrogate_rng", np.random.default_rng(0))
    started_at = time.perf_counter()

    for sample_index, transition in enumerate(selected_transitions):
        step = steps_by_origin.get(int(transition.origin_index))
        if step is None:
            raise RuntimeError("full-information Cache evaluation lacks current aligned step")
        next_step = steps_by_origin.get(int(transition.origin_index) + 1)
        if not bool(transition.done) and next_step is None:
            raise RuntimeError("full-information Cache evaluation lacks next aligned step")
        branch_cache: dict[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]], float
        ] = {}
        branch_component_cache: dict[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]], dict[str, float]
        ] = {}
        full_exact_branch_keys: set[
            tuple[int, bytes, tuple[tuple[str, float, float], ...]]
        ] = set()
        for node_index in node_indices:
            state = _node_policy_state(
                trainer,
                transition,
                step,
                node_index,
                feature_dim,
            )
            action_map = _production_action_map(transition)
            sequential_triggers = np.zeros(node_count, dtype=np.int8)
            for earlier_index in range(node_index):
                if (
                    str(action_map[earlier_index].get("conditional_action", "")).upper()
                    != "KEEP"
                ):
                    sequential_triggers[earlier_index] = 1
            sequential_triggers[node_index] = 1
            proposal_cache_nodes = uav_configuration.cache_nodes_with_uav_positions(
                trainer.context.cache_nodes,
                _position_dict(transition.previous_uav_positions),
            )
            universe_candidates_list, universe_features = (
                _all_cache_target_configurations_with_features(
                    state["working_cache"],
                    state["previous_cache"],
                    sequential_triggers,
                    node_index,
                    step,
                    trainer.context,
                    proposal_cache_nodes,
                )
            )
            universe_candidates = tuple(universe_candidates_list)
            universe_count = len(universe_candidates)
            feasible_universe_counts.append(universe_count)
            logical_candidate_count += universe_count
            keep_branch_count += 1
            trigger_policy_probabilities[sample_index, node_index] = (
                float(state["p_keep"]),
                float(state["p_update"]),
            )
            for candidate in (None, *universe_candidates):
                full_exact_branch_keys.add(
                    _cache_branch_identity(
                        transition,
                        node_index,
                        state,
                        candidate,
                    )
                )

            predicted_delta = np.zeros(universe_count, dtype=np.float64)
            predicted_sigma = np.zeros(universe_count, dtype=np.float64)
            if (
                universe_count
                and not force_full_exact
                and getattr(trainer, "cache_surrogate_ensemble", None) is not None
            ):
                inference_started = time.perf_counter()
                predicted_delta, predicted_sigma = trainer._predict_cache_surrogate(
                    np.asarray(state["conditional_context"], dtype=np.float32),
                    universe_features,
                )
                surrogate_inference_runtime_seconds += time.perf_counter() - inference_started
                predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
                predicted_sigma = np.asarray(predicted_sigma, dtype=np.float64)
            periodic_audit = bool(
                surrogate_ready
                and universe_count
                and (int(transition.origin_index) - int(trainer.context.train_origin_start))
                % max(int(getattr(trainer, "cache_surrogate_audit_interval", 20)), 1)
                == 0
            )
            full_exact_row = bool(force_full_exact or not surrogate_ready or periodic_audit)
            production_selected_index = _selected_cache_candidate_index(
                transition,
                node_index,
                list(universe_candidates),
            )
            anchor_indices = tuple(
                index
                for index, candidate in enumerate(universe_candidates)
                if str(candidate.proposal_source)
                in {"prediction", "coverage_anchor", "intermediate"}
            )
            random_audit_index = (
                None
                if not universe_count
                else int(surrogate_rng.integers(0, universe_count))
            )
            if full_exact_row:
                exact_indices = list(range(universe_count))
                if periodic_audit:
                    full_exact_audit_count += 1
            else:
                selected_k = max(
                    1,
                    min(
                        int(getattr(trainer, "cache_surrogate_selected_k", None) or 1),
                        universe_count,
                    ),
                )
                exact_indices = _hybrid_exact_indices(
                    predicted_delta,
                    predicted_sigma,
                    selected_k,
                    anchor_indices=anchor_indices,
                    production_selected_index=production_selected_index,
                    random_audit_index=random_audit_index,
                )
            exact_candidate_count += len(exact_indices)
            surrogate_candidate_count += universe_count - len(exact_indices)
            branch_candidates: list[Any | None] = [None]
            branch_candidates.extend(universe_candidates[index] for index in exact_indices)
            exact_values: list[float] = []
            for candidate in branch_candidates:
                identity = _cache_branch_identity(
                    transition,
                    node_index,
                    state,
                    candidate,
                )
                q_value = branch_cache.get(identity)
                if q_value is None:
                    branch = _execute_cache_branch(
                        trainer,
                        transition,
                        step,
                        next_step,
                        node_index,
                        state,
                        candidate,
                    )
                    if include_q_component_diagnostics or include_full_candidate_q_components:
                        branch_components = _branch_q_components(trainer, [branch])
                        q_value = float(branch_components["branch_q"][0])
                        branch_component_cache[identity] = {
                            "reward": float(branch_components["branch_reward"][0]),
                            "next_v": float(
                                branch_components["branch_next_critic_value"][0]
                            ),
                            "q": q_value,
                        }
                        max_q_component_reconstruction_error = max(
                            max_q_component_reconstruction_error,
                            float(branch_components["max_reconstruction_error"]),
                        )
                    else:
                        q_value = float(_branch_q_values(trainer, [branch])[0])
                    branch_cache[identity] = q_value
                    environment_executions += 1
                else:
                    deduplicated_branches += 1
                exact_values.append(q_value)
            keep_value = float(exact_values[0])
            q_keep[sample_index, node_index] = keep_value
            if universe_count:
                hybrid_q = keep_value + predicted_delta
                exact_by_index = np.full(universe_count, np.nan, dtype=np.float64)
                if exact_indices:
                    exact_by_index[np.asarray(exact_indices, dtype=np.int64)] = np.asarray(
                        exact_values[1:], dtype=np.float64
                    )
                    hybrid_q[np.asarray(exact_indices, dtype=np.int64)] = exact_by_index[
                        np.asarray(exact_indices, dtype=np.int64)
                    ]
                # 最终精确确认：Hybrid 获胜候选必须经过精确评估，
                # 才能成为 best_candidate_index 或
                # q_best_update。Surrogate 值仍用于筛选和
                # 排序；若获胜候选只有 surrogate 结果，则会重新精确评估，
                # 并在混合价值向量上重新取最大值，直到获胜候选
                # 已精确评估。这样可确保获胜候选语义有效，但不保证
                # 它一定是全局精确最优项；从未经过精确评估的候选中
                # 仍可能存在更高的真实价值。
                confirmed_exact_mask = np.zeros(universe_count, dtype=bool)
                confirmed_exact_mask[np.asarray(exact_indices, dtype=np.int64)] = True
                confirmation_added = 0
                while True:
                    universe_best_index = int(np.argmax(hybrid_q))
                    if confirmed_exact_mask[universe_best_index]:
                        break
                    winner_candidate = universe_candidates[universe_best_index]
                    winner_identity = _cache_branch_identity(
                        transition,
                        node_index,
                        state,
                        winner_candidate,
                    )
                    winner_q = branch_cache.get(winner_identity)
                    if winner_q is None:
                        winner_branch = _execute_cache_branch(
                            trainer,
                            transition,
                            step,
                            next_step,
                            node_index,
                            state,
                            winner_candidate,
                        )
                        if include_q_component_diagnostics or include_full_candidate_q_components:
                            winner_components = _branch_q_components(
                                trainer, [winner_branch]
                            )
                            winner_q = float(winner_components["branch_q"][0])
                            branch_component_cache[winner_identity] = {
                                "reward": float(winner_components["branch_reward"][0]),
                                "next_v": float(
                                    winner_components[
                                        "branch_next_critic_value"
                                    ][0]
                                ),
                                "q": winner_q,
                            }
                            max_q_component_reconstruction_error = max(
                                max_q_component_reconstruction_error,
                                float(winner_components["max_reconstruction_error"]),
                            )
                        else:
                            winner_q = float(
                                _branch_q_values(trainer, [winner_branch])[0]
                            )
                        branch_cache[winner_identity] = winner_q
                        environment_executions += 1
                    else:
                        deduplicated_branches += 1
                    hybrid_q[universe_best_index] = winner_q
                    exact_by_index[universe_best_index] = winner_q
                    confirmed_exact_mask[universe_best_index] = True
                    confirmation_added += 1
                confirmation_added_exact_count += confirmation_added
                hybrid_delta = hybrid_q - keep_value
                required_indices = [universe_best_index]
                if production_selected_index is not None:
                    required_indices.append(int(production_selected_index))
                policy_candidates = cache_configuration.select_cache_target_configurations(
                    list(universe_candidates),
                    np.asarray(state["working_cache"], dtype=np.int8)[node_index],
                    candidate_capacity,
                    predicted_delta_q=hybrid_delta,
                    predicted_sigma=predicted_sigma,
                    diversity_reserve=min(12, max(0, candidate_capacity - 1)),
                    required_indices=tuple(required_indices),
                )
                universe_index_by_configuration = {
                    np.ascontiguousarray(candidate.target_node_cache).tobytes(): index
                    for index, candidate in enumerate(universe_candidates)
                }
                policy_indices = np.asarray(
                    [
                        universe_index_by_configuration[
                            np.ascontiguousarray(candidate.target_node_cache).tobytes()
                        ]
                        for candidate in policy_candidates
                    ],
                    dtype=np.int64,
                )
                if include_full_candidate_q_components:
                    full_candidate_q_components.append(
                        {
                            "transition_index": int(selected_indices[sample_index]),
                            "origin_index": int(transition.origin_index),
                            "head_index": int(node_index),
                            "candidates": [
                                {
                                    "universe_index": int(candidate_index),
                                    "target_node_cache": np.asarray(
                                        universe_candidates[int(candidate_index)].target_node_cache,
                                        dtype=np.int8,
                                    ).astype(int).tolist(),
                                    **dict(
                                        branch_component_cache[
                                            _cache_branch_identity(
                                                transition,
                                                node_index,
                                                state,
                                                universe_candidates[int(candidate_index)],
                                            )
                                        ]
                                    ),
                                }
                                for candidate_index in range(universe_count)
                            ],
                        }
                    )
                count = len(policy_indices)
                if count > candidate_capacity:
                    raise RuntimeError("post-Hybrid policy Candidate set exceeds capacity")
                candidate_counts[sample_index, node_index] = count
                policy_best_recalled.append(float(universe_best_index in set(policy_indices)))
                candidate_features[sample_index, node_index, :count] = universe_features[
                    policy_indices
                ]
                candidate_operation_indices[sample_index, node_index, :count] = 0
                candidate_policy_probabilities[sample_index, node_index, :count] = (
                    _candidate_probabilities(
                        trainer.policy,
                        np.asarray(state["conditional_context"], dtype=np.float32),
                        universe_features[policy_indices],
                    )
                )
                candidate_valid[sample_index, node_index, :count] = True
                candidate_q[sample_index, node_index, :count] = hybrid_q[policy_indices]
                candidate_predicted_delta_q[sample_index, node_index, :count] = predicted_delta[
                    policy_indices
                ]
                candidate_predicted_sigma[sample_index, node_index, :count] = predicted_sigma[
                    policy_indices
                ]
                policy_exact = np.isfinite(exact_by_index[policy_indices])
                candidate_exact_evaluated[sample_index, node_index, :count] = policy_exact
                candidate_q_sources[sample_index, node_index, :count] = np.where(
                    policy_exact, "EXACT_Q", "SURROGATE_Q"
                )
                exact_target_indices = np.asarray(exact_indices, dtype=np.int64)
                for candidate_index in exact_target_indices:
                    exact_context_labels.append(
                        np.asarray(state["conditional_context"], dtype=np.float32)
                    )
                    exact_feature_labels.append(
                        np.asarray(universe_features[int(candidate_index)], dtype=np.float32)
                    )
                    exact_delta_labels.append(
                        float(exact_by_index[int(candidate_index)] - keep_value)
                    )
                    exact_group_labels.append(
                        int(int(transition.origin_index) * node_count + node_index)
                    )
                if full_exact_row:
                    exact_delta = exact_by_index - keep_value
                    screening_rows.append(
                        {
                            "predicted_delta_q": predicted_delta.copy(),
                            "predicted_sigma": predicted_sigma.copy(),
                            "exact_q": exact_delta.copy(),
                            "anchor_indices": anchor_indices,
                            "production_selected_index": production_selected_index,
                            "random_audit_index": random_audit_index,
                        }
                    )
                if include_q_component_diagnostics:
                    keep_identity = _cache_branch_identity(
                        transition, node_index, state, None
                    )
                    keep_components = branch_component_cache.get(keep_identity)
                    if keep_components is None:
                        raise RuntimeError(
                            "Cache KEEP Exact-Q component diagnostics are missing"
                        )
                    winner_components = branch_component_cache.get(
                        _cache_branch_identity(
                            transition,
                            node_index,
                            state,
                            universe_candidates[universe_best_index],
                        )
                    )
                    winner_policy_index = int(
                        np.flatnonzero(policy_indices == universe_best_index)[0]
                    )
                    if winner_components is None:
                        raise RuntimeError(
                            "Cache confirmed-winner Exact-Q components are missing"
                        )
                    q_component_diagnostics[sample_index].append(
                        {
                            "head_index": int(node_index),
                            "head_id": str(
                                trainer.context.action_spec.cache_node_ids[node_index]
                            ),
                            "keep": dict(keep_components),
                            "final_exact_confirmed_winner": {
                                **dict(winner_components),
                                "policy_candidate_index": winner_policy_index,
                                "universe_candidate_index": int(universe_best_index),
                                "target_node_cache": np.asarray(
                                    universe_candidates[universe_best_index].target_node_cache,
                                    dtype=np.int8,
                                ).astype(int).tolist(),
                                "proposal_source": str(
                                    universe_candidates[universe_best_index].proposal_source
                                ),
                            },
                            "critic_provenance_reference": "critic_parameter_digest_before_teacher",
                        }
                    )
            elif include_q_component_diagnostics:
                keep_identity = _cache_branch_identity(
                    transition, node_index, state, None
                )
                keep_components = branch_component_cache.get(keep_identity)
                if keep_components is None:
                    raise RuntimeError(
                        "Cache KEEP Exact-Q component diagnostics are missing"
                    )
                q_component_diagnostics[sample_index].append(
                    {
                        "head_index": int(node_index),
                        "head_id": str(
                            trainer.context.action_spec.cache_node_ids[node_index]
                        ),
                        "keep": dict(keep_components),
                        "final_exact_confirmed_winner": None,
                        "critic_provenance_reference": "critic_parameter_digest_before_teacher",
                    }
                )
        full_exact_environment_baseline += len(full_exact_branch_keys)

    q_best = np.full((sample_count, node_count), np.nan, dtype=np.float64)
    delta_best = np.full_like(q_best, np.nan)
    best_index = np.full((sample_count, node_count), -1, dtype=np.int64)
    trigger_teacher = np.zeros((sample_count, node_count, 2), dtype=np.float64)
    trigger_teacher[..., 0] = 1.0
    candidate_teacher = np.zeros_like(candidate_q)
    trigger_valid = candidate_valid.any(axis=-1)
    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
            legal_q = candidate_q[sample_index, head_index, legal]
            legal_indices = np.flatnonzero(legal)
            selected = int(legal_indices[int(np.argmax(legal_q))])
            best_index[sample_index, head_index] = selected
            q_best[sample_index, head_index] = float(
                candidate_q[sample_index, head_index, selected]
            )
            delta_best[sample_index, head_index] = (
                q_best[sample_index, head_index] - q_keep[sample_index, head_index]
            )
    trigger_deltas = delta_best[np.isfinite(delta_best)]
    candidate_ranges: list[float] = []
    candidate_stds: list[float] = []
    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
            row = candidate_q[sample_index, head_index, legal]
            candidate_ranges.append(float(np.max(row) - np.min(row)))
            candidate_stds.append(float(np.std(row)))

    scale_epsilon = float(getattr(trainer, "cache_teacher_scale_epsilon", 1.0e-3))
    scale_maximum = float(getattr(trainer, "cache_teacher_scale_max", 1.0e3))
    calibration = dict(getattr(trainer, "cache_teacher_scale_calibration", {}) or {})

    def resolve_scale(
        field: str,
        supplied: float | None,
        values: np.ndarray,
    ) -> float:
        if supplied is not None:
            return float(supplied)
        batch_scale = _robust_mad_scale(values, scale_epsilon, scale_maximum)
        previous = calibration.get(field)
        if previous is None or not math.isfinite(float(previous)) or float(previous) <= 0.0:
            resolved = batch_scale
        else:
            previous_scale = float(previous)
            clipped_batch = float(
                np.clip(batch_scale, previous_scale * 0.5, previous_scale * 2.0)
            )
            resolved = float(np.clip(
                0.5 * previous_scale + 0.5 * clipped_batch,
                scale_epsilon,
                scale_maximum,
            ))
        calibration[field] = resolved
        calibration[field + "_latest_batch_mad_scale"] = batch_scale
        calibration[field + "_source"] = "TRAIN_ROLLOUT_MAD"
        return resolved

    trigger_scale = resolve_scale("trigger_temperature", trigger_scale, trigger_deltas)
    if trigger_temperature is None:
        calibration.update(
            {
                "method": "CLIPPED_RUNNING_MAD",
                "source": "TRAIN_ROLLOUT_ONLY",
                "epsilon": scale_epsilon,
                "maximum": scale_maximum,
                "training_value_count": int(len(trigger_deltas)),
            }
        )
        trainer.cache_teacher_scale_calibration = calibration
        trainer.cache_trigger_teacher_temperature = float(trigger_scale)
    trainer.cache_candidate_teacher_temperature = 1.0

    candidate_group_diagnostics: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        for head_index in node_indices:
            legal = candidate_valid[sample_index, head_index]
            if not np.any(legal):
                continue
            trigger_teacher[sample_index, head_index] = _stable_masked_softmax(
                np.asarray(
                    [q_keep[sample_index, head_index], q_best[sample_index, head_index]],
                    dtype=np.float64,
                ),
                np.asarray([True, True], dtype=bool),
                trigger_scale,
            )
            candidate_row, candidate_diagnostics = state_head_candidate_teacher(
                candidate_q[sample_index, head_index],
                legal,
                scale_epsilon,
            )
            candidate_teacher[sample_index, head_index] = candidate_row
            candidate_diagnostics.update(
                {
                    "sample_index": int(sample_index),
                    "transition_index": int(selected_indices[sample_index]),
                    "origin_index": int(
                        selected_transitions[sample_index].origin_index
                    ),
                    "head_index": int(head_index),
                    "head_id": str(
                        trainer.context.action_spec.cache_node_ids[head_index]
                    ),
                    "role": str(trainer.context.cache_nodes["node_roles"][head_index]),
                }
            )
            candidate_group_diagnostics.append(candidate_diagnostics)
    if not np.all(np.isfinite(q_keep[:, node_indices])):
        raise FloatingPointError("full-exact Cache KEEP values are non-finite")
    if not np.all(np.isfinite(candidate_q[candidate_valid])):
        raise FloatingPointError("Cache candidate values are non-finite")
    surrogate_train_started = time.perf_counter()
    if exact_delta_labels and train_surrogate and not force_full_exact:
        surrogate_training_metrics = trainer._train_cache_surrogate(
            np.asarray(exact_context_labels, dtype=np.float32).reshape(
                -1, int(trainer.cache_surrogate_context_dim)
            ),
                np.asarray(exact_feature_labels, dtype=np.float32).reshape(
                    -1, int(trainer.cache_surrogate_feature_dim)
                ),
                np.asarray(exact_delta_labels, dtype=np.float32),
                ranking_group_ids=np.asarray(exact_group_labels, dtype=np.int64),
            )
    surrogate_training_runtime_seconds = float(
        time.perf_counter() - surrogate_train_started
    )
    gate_metrics = _screening_gate_metrics(screening_rows) if screening_rows else None
    delta_residuals = [
        float(predicted - exact)
        for row in screening_rows
        for predicted, exact in zip(row["predicted_delta_q"], row["exact_q"])
    ]
    delta_q_mae = (
        float(np.mean(np.abs(delta_residuals))) if delta_residuals else math.nan
    )
    delta_q_rmse = (
        float(np.sqrt(np.mean(np.square(delta_residuals))))
        if delta_residuals
        else math.nan
    )
    trigger_scale_stats = _cache_teacher_probability_stats(
        trigger_teacher,
        trigger_valid,
        trigger=True,
    )
    candidate_scale_stats = _cache_teacher_probability_stats(
        candidate_teacher,
        candidate_valid,
        trigger=False,
    )
    trigger_delta_stats = _audit_value_stats(trigger_deltas.tolist())
    trigger_delta_median = float(np.median(trigger_deltas)) if len(trigger_deltas) else 0.0
    trigger_delta_mad = (
        float(np.median(np.abs(trigger_deltas - trigger_delta_median)))
        if len(trigger_deltas)
        else 0.0
    )
    trigger_delta_stats.update(
        {
            "std": float(np.std(trigger_deltas)) if len(trigger_deltas) else 0.0,
            "mad": trigger_delta_mad,
            "p10": float(np.percentile(trigger_deltas, 10.0)) if len(trigger_deltas) else 0.0,
            "p25": float(np.percentile(trigger_deltas, 25.0)) if len(trigger_deltas) else 0.0,
            "p50": trigger_delta_median,
            "p75": float(np.percentile(trigger_deltas, 75.0)) if len(trigger_deltas) else 0.0,
            "p90": float(np.percentile(trigger_deltas, 90.0)) if len(trigger_deltas) else 0.0,
            "abs_p50": float(np.percentile(np.abs(trigger_deltas), 50.0)) if len(trigger_deltas) else 0.0,
            "abs_p90": float(np.percentile(np.abs(trigger_deltas), 90.0)) if len(trigger_deltas) else 0.0,
            "abs_p95": float(np.percentile(np.abs(trigger_deltas), 95.0)) if len(trigger_deltas) else 0.0,
        }
    )
    return {
        "q_keep": q_keep.astype(np.float32),
        "candidate_q_values": candidate_q.astype(np.float32),
        "candidate_valid_mask": candidate_valid,
        "best_candidate_index": best_index,
        "q_best_update": q_best.astype(np.float32),
        "delta_best": delta_best.astype(np.float32),
        "trigger_teacher_probabilities": trigger_teacher.astype(np.float32),
        "trigger_teacher_valid_mask": trigger_valid,
        "candidate_teacher_probabilities": candidate_teacher.astype(np.float32),
        "candidate_policy_probabilities": candidate_policy_probabilities.astype(np.float32),
        "candidate_predicted_delta_q": candidate_predicted_delta_q.astype(np.float32),
        "candidate_predicted_sigma": candidate_predicted_sigma.astype(np.float32),
        "candidate_exact_evaluated_mask": candidate_exact_evaluated,
        "candidate_q_sources": candidate_q_sources,
        "trigger_policy_probabilities": trigger_policy_probabilities.astype(np.float32),
        "candidate_features": candidate_features,
        "candidate_operation_indices": candidate_operation_indices,
        "candidate_counts": candidate_counts,
        "feasible_universe_counts": np.asarray(feasible_universe_counts, dtype=np.int64),
        "feasible_universe_enumeration_coverage": 1.0,
        "full_universe_candidate_count_mean": (
            float(np.mean(feasible_universe_counts)) if feasible_universe_counts else 0.0
        ),
        "full_universe_candidate_count_p90": (
            float(np.percentile(feasible_universe_counts, 90.0))
            if feasible_universe_counts
            else 0.0
        ),
        "full_universe_candidate_count_max": (
            int(np.max(feasible_universe_counts)) if feasible_universe_counts else 0
        ),
        "policy_candidate_best_recall": (
            float(np.mean(policy_best_recalled)) if policy_best_recalled else 1.0
        ),
        "counterfactual_environment_executions": int(environment_executions),
        "logical_candidate_count": int(logical_candidate_count),
        "keep_branch_count": int(keep_branch_count),
        "deduplicated_branch_count": int(deduplicated_branches),
        "runtime_seconds": float(time.perf_counter() - started_at),
        "full_exact_environment_baseline": int(full_exact_environment_baseline),
        "exact_candidate_count": int(exact_candidate_count),
        "confirmation_added_exact_count": int(confirmation_added_exact_count),
        "surrogate_candidate_count": int(surrogate_candidate_count),
        "full_exact_audit_count": int(full_exact_audit_count),
        "surrogate_inference_runtime_seconds": float(surrogate_inference_runtime_seconds),
        "surrogate_training_runtime_seconds": surrogate_training_runtime_seconds,
        "surrogate_training": surrogate_training_metrics,
        "surrogate_selected_k": getattr(trainer, "cache_surrogate_selected_k", None),
        "surrogate_gate_metrics": gate_metrics,
        "surrogate_delta_q_mae": delta_q_mae,
        "surrogate_delta_q_rmse": delta_q_rmse,
        "surrogate_screening_active": bool(surrogate_ready),
        "full_exact_evaluation": bool(gate_metrics is not None),
        "cache_teacher_scale_calibration": calibration,
        "trigger_delta_best_statistics": trigger_delta_stats,
        "candidate_q_range_mean": float(np.mean(candidate_ranges)) if candidate_ranges else 0.0,
        "candidate_q_std_mean": float(np.mean(candidate_stds)) if candidate_stds else 0.0,
        "trigger_teacher_statistics": trigger_scale_stats,
        "candidate_teacher_statistics": candidate_scale_stats,
        "candidate_teacher_group_diagnostics": candidate_group_diagnostics,
        "candidate_teacher_scale_mode": "STATE_HEAD_ROBUST_NORMALIZATION",
        "candidate_teacher_center": "MEDIAN",
        "candidate_teacher_scale": "MAD_1P4826_WITH_STD_RANGE_EPS_FALLBACK",
        "value_evaluator": (
            "HYBRID_EXACT_SURROGATE"
            if surrogate_candidate_count > 0
            else "FULL_EXACT"
        ),
        "downstream_mode": "FIXED_DOWNSTREAM",
        "trigger_temperature": trigger_scale,
        "candidate_temperature": candidate_scale,
        "q_component_diagnostics": (
            q_component_diagnostics if include_q_component_diagnostics else None
        ),
        "full_candidate_q_components": (
            full_candidate_q_components
            if include_full_candidate_q_components
            else None
        ),
        "max_q_component_reconstruction_error": (
            float(max_q_component_reconstruction_error)
            if include_q_component_diagnostics or include_full_candidate_q_components
            else None
        ),
    }


def _audit_value_stats(values: list[float]) -> dict[str, float | None]:
    """返回 teacher 价值对应的紧凑有限分布统计量。"""

    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("Cache teacher audit statistics are non-finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }

def _cache_per_head_old_new_kl(
    old_operation_log_probs: Any,
    new_operation_log_probs: Any,
    old_candidate_log_probs: Any,
    new_candidate_log_probs: Any,
    operation_masks: Any,
    candidate_masks: Any,
) -> Any:
    """返回每个 Cache head 从旧策略到新策略的 KL。每个 head 的动作集合包含 KEEP 和所有合法 UPDATE 候选；UPDATE 概率由 Trigger 概率与条件候选概率相乘得到，与 Stage2A 损失使用的原子动作 logp 一致。mask 固定使用 rollout 数据，padding 候选的贡献为零。"""

    if torch is None:  # pragma: no cover - Trainer 构造时已检查依赖
        raise RuntimeError("Cache per-head KL requires torch")
    old_operation = torch.as_tensor(old_operation_log_probs)
    new_operation = torch.as_tensor(
        new_operation_log_probs,
        dtype=old_operation.dtype,
        device=old_operation.device,
    )
    old_candidate = torch.as_tensor(
        old_candidate_log_probs,
        dtype=old_operation.dtype,
        device=old_operation.device,
    )
    new_candidate = torch.as_tensor(
        new_candidate_log_probs,
        dtype=old_operation.dtype,
        device=old_operation.device,
    )
    operation_mask = torch.as_tensor(
        operation_masks,
        dtype=torch.bool,
        device=old_operation.device,
    )
    candidate_mask = torch.as_tensor(
        candidate_masks,
        dtype=torch.bool,
        device=old_operation.device,
    )
    if old_operation.ndim != 3 or int(old_operation.shape[-1]) != 2:
        raise ValueError("Cache operation log-probabilities must be [B,H,2]")
    if tuple(new_operation.shape) != tuple(old_operation.shape):
        raise ValueError("Cache old/new operation log-probability shape mismatch")
    if old_candidate.ndim != 3 or tuple(new_candidate.shape) != tuple(old_candidate.shape):
        raise ValueError("Cache old/new candidate log-probability shape mismatch")
    if tuple(operation_mask.shape) != tuple(old_operation.shape):
        raise ValueError("Cache operation mask shape mismatch")
    if tuple(candidate_mask.shape) != tuple(old_candidate.shape):
        raise ValueError("Cache candidate mask shape mismatch")
    if not bool(torch.all(torch.any(operation_mask, dim=-1)).item()):
        raise ValueError("Cache operation mask blocks every action")
    if not bool(torch.all(torch.any(candidate_mask, dim=-1)).item()):
        raise ValueError("Cache candidate mask blocks every candidate")
    for values, mask, label in (
        (old_operation, operation_mask, "old operation"),
        (new_operation, operation_mask, "new operation"),
        (old_candidate, candidate_mask, "old candidate"),
        (new_candidate, candidate_mask, "new candidate"),
    ):
        if not bool(torch.all(torch.isfinite(values[mask])).item()):
            raise FloatingPointError(f"Cache {label} log-probabilities are non-finite")

    safe_old_operation = torch.where(
        operation_mask, old_operation, torch.zeros_like(old_operation)
    )
    safe_new_operation = torch.where(
        operation_mask, new_operation, torch.zeros_like(new_operation)
    )
    safe_old_candidate = torch.where(
        candidate_mask, old_candidate, torch.zeros_like(old_candidate)
    )
    safe_new_candidate = torch.where(
        candidate_mask, new_candidate, torch.zeros_like(new_candidate)
    )
    old_operation_probabilities = torch.where(
        operation_mask,
        torch.exp(safe_old_operation),
        torch.zeros_like(safe_old_operation),
    )
    old_candidate_probabilities = torch.where(
        candidate_mask,
        torch.exp(safe_old_candidate),
        torch.zeros_like(safe_old_candidate),
    )
    old_keep = old_operation_probabilities[..., 0]
    old_update = old_operation_probabilities[..., 1]
    keep_kl = old_keep * (
        safe_old_operation[..., 0] - safe_new_operation[..., 0]
    )
    update_log_delta = (
        safe_old_operation[..., 1].unsqueeze(-1)
        + safe_old_candidate
        - safe_new_operation[..., 1].unsqueeze(-1)
        - safe_new_candidate
    )
    update_kl = (
        old_update.unsqueeze(-1)
        * old_candidate_probabilities
        * update_log_delta
    ).sum(dim=-1)
    return torch.clamp(keep_kl + update_kl, min=0.0)

def _candidate_probability_log_distribution(
    probabilities: Any,
    candidate_valid_mask: Any,
) -> tuple[Any, Any]:
    """返回归一化候选 log probability 和安全的 KL mask。没有合法 UPDATE 候选的状态仍有定义良好的 Cache 动作分布，此时 KEEP 是唯一动作。Candidate 策略在这些行上返回全零概率，因此 KL 计算中只为 UPDATE 条件分布加入带 mask 的单位质量哨兵值。"""

    if torch is None:  # pragma: no cover - Trainer 构造时已检查依赖
        raise RuntimeError("candidate distributions require torch")
    probability_t = torch.as_tensor(probabilities)
    valid_t = torch.as_tensor(
        candidate_valid_mask,
        dtype=torch.bool,
        device=probability_t.device,
    )
    if probability_t.ndim != 3 or tuple(valid_t.shape) != tuple(probability_t.shape):
        raise ValueError("Cache candidate probability/mask shape mismatch")
    if int(probability_t.shape[-1]) < 1:
        raise ValueError("Cache candidate distributions require a padded candidate")
    active = valid_t.any(dim=-1)
    fallback = torch.zeros_like(valid_t)
    fallback[..., 0] = ~active
    kl_mask = valid_t | fallback
    safe_probabilities = torch.where(
        valid_t, probability_t, torch.zeros_like(probability_t)
    )
    fallback_probabilities = torch.zeros_like(probability_t)
    fallback_probabilities[..., 0] = 1.0
    safe_probabilities = torch.where(
        active.unsqueeze(-1), safe_probabilities, fallback_probabilities
    )
    if not bool(torch.all(torch.isfinite(safe_probabilities[kl_mask])).item()):
        raise FloatingPointError("Cache candidate probabilities are non-finite")
    log_probabilities = torch.where(
        kl_mask,
        torch.log(safe_probabilities.clamp_min(1.0e-12)),
        torch.zeros_like(safe_probabilities),
    )
    return log_probabilities, kl_mask
