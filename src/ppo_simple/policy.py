"""PPO v2 trigger policy and adaptive primitive Cache/UAV policy schema."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PolicySample:
    action: np.ndarray
    log_prob: Any
    entropy: Any
    value: Any
    action_mask: np.ndarray
    action_probabilities: np.ndarray | None = None
    deterministic_action: np.ndarray | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: Any
    entropy: Any
    value: Any
    logits: Any
    # Optional branch-level diagnostics.  These are populated only by policy
    # schemas whose rollout already carries the corresponding primitive
    # actions and old log-probabilities; they do not affect the PPO objective.
    cache_log_prob: Any | None = None
    uav_log_prob: Any | None = None
    # Per-UAV operation/candidate log-probabilities.  The aggregate branch
    # values above remain the compatibility surface for the existing PPO
    # schemas; R2 uses these tensors to keep the operation trust region
    # independent from the candidate trust region.
    uav_operation_log_probs: Any | None = None
    uav_candidate_log_probs: Any | None = None
    # Full conditional UAV candidate log-probabilities.  This is populated by
    # the Recovery R4 evaluator so the exact conditional PPO expectation can
    # enumerate every legal MOVE candidate, including states whose sampled
    # operation was STAY.
    uav_candidate_log_probs_all: Any | None = None
    cache_entropy: Any | None = None
    uav_entropy: Any | None = None
    # Exact per-Cache-node primitive log-probabilities.  These include the
    # masked operation categorical and, when the operation is not NO_CHANGE,
    # the selected candidate categorical.
    cache_node_log_probs: Any | None = None


@dataclass(frozen=True)
class CandidatePolicySample:
    selected_index: int
    log_prob: Any
    entropy: Any
    logits: Any


@dataclass(frozen=True)
class OperationPolicySample:
    selected_index: int
    log_prob: Any
    entropy: Any
    logits: Any
    # Optional raw (pre-mask) diagnostics. These are derived from the
    # already-computed operation logits and never sample an additional action.
    raw_logits: Any | None = None
    probabilities: Any | None = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PPO v2 requires PyTorch.")


def _validate_binary_actions(actions: Any, action_dim: int) -> Any:
    values = torch.as_tensor(actions, dtype=torch.long)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or tuple(values.shape[1:]) != (int(action_dim),):
        raise ValueError(
            f"actions shape={tuple(values.shape)}，期望 [B,{int(action_dim)}]。"
        )
    if not bool(torch.all((values == 0) | (values == 1)).item()):
        raise ValueError("PPO v2 action 必须是 binary request vector。")
    return values


def _normalize_masks(action_masks: Any, batch_size: int, action_dim: int, device: Any) -> Any:
    if action_masks is None:
        return torch.ones(
            (int(batch_size), int(action_dim), 2),
            dtype=torch.bool,
            device=device,
        )
    masks = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    expected = (int(batch_size), int(action_dim), 2)
    if tuple(masks.shape) != expected:
        raise ValueError(f"action_masks shape={tuple(masks.shape)}，期望 {expected}。")
    if not bool(torch.all(torch.any(masks, dim=-1)).item()):
        raise ValueError("action mask 不能同时屏蔽 binary request 的 0/1。")
    return masks


def _group_balance_candidate_logits(
    raw_scores: Any,
    candidate_features: Any,
    candidate_masks: Any,
    operation_slice: slice,
    active_heads: Any | None = None,
) -> Any:
    """Return masked logits with equal-score probability balanced by operation group."""

    scores = torch.as_tensor(raw_scores)
    features = torch.as_tensor(
        candidate_features,
        dtype=torch.float32,
        device=scores.device,
    )
    if scores.ndim < 1 or features.ndim != scores.ndim + 1:
        raise ValueError("raw_scores/candidate_features 维度不匹配。")
    if tuple(features.shape[:-1]) != tuple(scores.shape):
        raise ValueError("candidate_features 必须与 raw_scores 对齐。")
    candidate_count = int(scores.shape[-1])
    start = 0 if operation_slice.start is None else int(operation_slice.start)
    stop = int(features.shape[-1]) if operation_slice.stop is None else int(operation_slice.stop)
    if start < 0 or stop <= start or stop > int(features.shape[-1]):
        raise ValueError("operation_slice 超出 candidate feature 范围。")
    masks = (
        torch.ones_like(scores, dtype=torch.bool)
        if candidate_masks is None
        else torch.as_tensor(candidate_masks, dtype=torch.bool, device=scores.device)
    )
    if tuple(masks.shape) != tuple(scores.shape):
        raise ValueError("candidate_masks 必须与 raw_scores 对齐。")
    if not bool(torch.all(torch.any(masks, dim=-1)).item()):
        raise ValueError("每个 conditional head 至少需要一个 valid candidate。")
    if not bool(torch.all(torch.isfinite(scores)).item()):
        raise ValueError("candidate scores 包含 NaN/Inf。")
    if active_heads is None:
        active = torch.ones(scores.shape[:-1], dtype=torch.bool, device=scores.device)
    else:
        active = torch.as_tensor(active_heads, dtype=torch.bool, device=scores.device)
        if tuple(active.shape) != tuple(scores.shape[:-1]):
            raise ValueError("active_heads 必须与 candidate head 维度对齐。")

    flat_scores = scores.reshape(-1, candidate_count)
    flat_features = features.reshape(-1, candidate_count, int(features.shape[-1]))
    flat_masks = masks.reshape(-1, candidate_count)
    flat_active = active.reshape(-1)
    operation_values = flat_features[..., start:stop]
    if not bool(torch.all(torch.isfinite(operation_values)).item()):
        raise ValueError("candidate operation feature 包含 NaN/Inf。")
    group_ids = torch.argmax(operation_values, dim=-1)
    flat_balanced = flat_scores.clone()
    for row_index in range(int(flat_scores.shape[0])):
        if not bool(flat_active[row_index].item()):
            continue
        valid_operations = operation_values[row_index][flat_masks[row_index]]
        if valid_operations.ndim != 2 or not int(valid_operations.shape[0]):
            raise ValueError("active head 没有 valid operation candidate。")
        if not bool(torch.all((valid_operations == 0.0) | (valid_operations == 1.0)).item()):
            raise ValueError("active candidate operation 必须是 one-hot。")
        if not bool(torch.all(valid_operations.sum(dim=-1) == 1.0).item()):
            raise ValueError("active candidate 必须属于唯一 operation group。")
        row_groups = group_ids[row_index]
        group_counts = torch.zeros(
            (stop - start,), dtype=torch.long, device=scores.device
        )
        group_counts.scatter_add_(
            0,
            row_groups[flat_masks[row_index]],
            torch.ones_like(row_groups[flat_masks[row_index]], dtype=torch.long),
        )
        valid_counts = group_counts.gather(0, row_groups)
        if bool(torch.any(valid_counts[flat_masks[row_index]] < 1).item()):
            raise ValueError("valid candidate 的 group_count 必须至少为1。")
        flat_balanced[row_index] = flat_scores[row_index] - torch.log(
            valid_counts.to(dtype=flat_scores.dtype)
        )
    return flat_balanced.reshape(scores.shape).masked_fill(~masks, -float("inf"))


class BernoulliActorCritic(nn.Module if nn is not None else object):
    """Dimension-driven MLP actor-critic with independent binary requests."""

    architecture_id = "ppo_simple_v2_mlp_independent_bernoulli"
    architecture_version = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        _require_torch()
        super().__init__()
        if int(observation_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("observation_dim/action_dim 必须大于0。")
        if not hidden_sizes or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes 必须是正整数列表。")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        layers: list[Any] = []
        last_dim = self.observation_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(last_dim, int(hidden_size)), nn.Tanh()])
            last_dim = int(hidden_size)
        self.encoder = nn.Sequential(*layers)
        self.hidden_dim = int(last_dim)
        self.actor_head = nn.Linear(last_dim, self.action_dim)
        self.critic_head = nn.Linear(last_dim, 1)

    def _observations(self, observations: Any) -> tuple[Any, bool]:
        device = next(self.parameters()).device
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        expected = (self.observation_dim,)
        if obs.ndim != 2 or tuple(obs.shape[1:]) != expected:
            raise ValueError(
                f"observations shape={tuple(obs.shape)}，期望 [B,{self.observation_dim}]。"
            )
        if not bool(torch.all(torch.isfinite(obs)).item()):
            raise ValueError("observations 包含 NaN/Inf。")
        return obs, single

    def _default_actor_encoder(self) -> Any:
        """Return the representation used by the legacy single-head path."""

        return self.encoder

    def forward(self, observations: Any) -> tuple[Any, Any]:
        obs, single = self._observations(observations)
        hidden = self._default_actor_encoder()(obs)
        logits = self.actor_head(hidden)
        values = self._critic_values(obs, hidden)
        if single:
            return logits.squeeze(0), values.squeeze(0)
        return logits, values

    def _critic_values(self, observations: Any, actor_hidden: Any) -> Any:
        del observations
        return self.critic_head(actor_hidden).squeeze(-1)

    def _masked_distribution(self, logits: Any, masks: Any) -> Any:
        binary_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return torch.distributions.Categorical(logits=binary_logits.masked_fill(~masks, -float("inf")))

    def sample(
        self,
        observation: Any,
        action_mask: Callable[[int, np.ndarray], Any] | Any = None,
        deterministic: bool = False,
        record_diagnostics: bool = False,
    ) -> PolicySample:
        logits, value = self.forward(observation)
        if logits.ndim != 1 or tuple(logits.shape) != (self.action_dim,):
            raise ValueError("sample 需要单个 observation。")
        actions: list[int] = []
        masks: list[np.ndarray] = []
        log_probs: list[Any] = []
        entropies: list[Any] = []
        action_probabilities: list[float] | None = [] if record_diagnostics else None
        deterministic_actions: list[int] | None = [] if record_diagnostics else None
        for action_index in range(self.action_dim):
            raw_mask = (
                np.ones(2, dtype=bool)
                if action_mask is None
                else action_mask(action_index, np.asarray(actions, dtype=np.int8))
            )
            mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
            if mask.shape != (2,) or not bool(np.any(mask)):
                raise ValueError(f"action_mask[{action_index}] 必须是 [2] 且至少允许一个值。")
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=logits.device).reshape(1, 2)
            distribution = self._masked_distribution(logits[action_index].reshape(1), mask_tensor)
            deterministic_index = int(torch.argmax(distribution.logits, dim=-1).item())
            value_index = (
                deterministic_index
                if deterministic
                else int(distribution.sample().item())
            )
            actions.append(value_index)
            masks.append(mask.copy())
            if action_probabilities is not None and deterministic_actions is not None:
                action_probabilities.append(
                    float(distribution.probs[0, 1].detach().cpu().item())
                )
                deterministic_actions.append(deterministic_index)
            action_tensor = torch.as_tensor([value_index], dtype=torch.long, device=logits.device)
            log_probs.append(distribution.log_prob(action_tensor).squeeze(0))
            entropies.append(distribution.entropy().squeeze(0))
        return PolicySample(
            action=np.asarray(actions, dtype=np.int8),
            log_prob=torch.stack(log_probs).sum(),
            entropy=torch.stack(entropies).sum(),
            value=value,
            action_mask=np.stack(masks, axis=0),
            action_probabilities=(
                None
                if action_probabilities is None
                else np.asarray(action_probabilities, dtype=np.float64)
            ),
            deterministic_action=(
                None
                if deterministic_actions is None
                else np.asarray(deterministic_actions, dtype=np.int8)
            ),
        )

    def evaluate(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any = None,
    ) -> PolicyEvaluation:
        obs, _ = self._observations(observations)
        hidden = self._default_actor_encoder()(obs)
        logits = self.actor_head(hidden)
        values = self._critic_values(obs, hidden)
        action_values = _validate_binary_actions(actions, self.action_dim).to(logits.device)
        masks = _normalize_masks(action_masks, int(obs.shape[0]), self.action_dim, logits.device)
        distribution = self._masked_distribution(logits, masks)
        valid = masks.gather(-1, action_values.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.all(valid).item()):
            raise ValueError("actions 包含被 action mask 屏蔽的 request。")
        return PolicyEvaluation(
            log_prob=distribution.log_prob(action_values).sum(dim=-1),
            entropy=torch.where(
                masks.sum(dim=-1) == 1,
                torch.zeros_like(distribution.entropy()),
                distribution.entropy(),
            ).sum(dim=-1),
            value=values,
            logits=logits,
        )


class AdaptivePrimitiveActorCritic(BernoulliActorCritic):
    """Shared-state PPO policy with trigger and conditional primitive heads."""

    architecture_id = "adaptive_primitive_v1"
    architecture_version = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(observation_dim, action_dim, hidden_sizes)
        if int(cache_feature_dim) <= 0 or int(uav_feature_dim) <= 0:
            raise ValueError("primitive candidate feature dimensions 必须大于0。")
        self.cache_feature_dim = int(cache_feature_dim)
        self.uav_feature_dim = int(uav_feature_dim)
        candidate_hidden = max(32, self.hidden_dim // 2)
        self.cache_candidate_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.cache_feature_dim, candidate_hidden),
            nn.Tanh(),
            nn.Linear(candidate_hidden, 1),
        )
        self.uav_candidate_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.uav_feature_dim, candidate_hidden),
            nn.Tanh(),
            nn.Linear(candidate_hidden, 1),
        )

    def _candidate_encoder_for_scorer(self, scorer: Any) -> Any:
        del scorer
        return self._default_actor_encoder()

    def _candidate_scores(
        self,
        observations: Any,
        candidate_features: Any,
        feature_dim: int,
        scorer: Any,
    ) -> tuple[Any, bool]:
        obs, single = self._observations(observations)
        features = torch.as_tensor(candidate_features, dtype=torch.float32, device=obs.device)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        expected = (int(obs.shape[0]), int(feature_dim))
        if features.ndim != 3 or int(features.shape[0]) != expected[0] or int(
            features.shape[2]
        ) != expected[1]:
            raise ValueError(
                "candidate_features shape 必须为 "
                f"[B,C,{int(feature_dim)}]，实际 {tuple(features.shape)}。"
            )
        if not bool(torch.all(torch.isfinite(features)).item()):
            raise ValueError("candidate_features 包含 NaN/Inf。")
        hidden = self._candidate_encoder_for_scorer(scorer)(obs)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, int(features.shape[1]), -1)
        scores = scorer(torch.cat([expanded_hidden, features], dim=-1)).squeeze(-1)
        return scores, single

    def _candidate_operation_slice(self, candidate_kind: str) -> slice | None:
        del candidate_kind
        return None

    def _sample_candidate(
        self,
        observation: Any,
        candidate_features: Any,
        feature_dim: int,
        scorer: Any,
        deterministic: bool,
        operation_slice: slice | None = None,
    ) -> CandidatePolicySample:
        scores, single = self._candidate_scores(
            observation,
            candidate_features,
            feature_dim,
            scorer,
        )
        if not single or scores.ndim != 2 or int(scores.shape[0]) != 1:
            raise ValueError("conditional candidate sampling 需要单个 context。")
        distribution_logits = scores.squeeze(0)
        if operation_slice is not None:
            distribution_logits = _group_balance_candidate_logits(
                distribution_logits,
                candidate_features,
                None,
                operation_slice,
            )
        distribution = torch.distributions.Categorical(logits=distribution_logits)
        selected = (
            int(torch.argmax(distribution.logits).item())
            if deterministic
            else int(distribution.sample().item())
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
        return CandidatePolicySample(
            selected_index=selected,
            log_prob=distribution.log_prob(selected_tensor),
            entropy=distribution.entropy(),
            logits=distribution_logits,
        )

    def _sample_candidate_grouped_deterministic(
        self,
        observation: Any,
        candidate_features: Any,
        feature_dim: int,
        scorer: Any,
        operation_slice: slice | None,
    ) -> CandidatePolicySample:
        """Select a candidate by group-MAP followed by within-group MAP.

        This is an evaluation-only decoder.  It reuses the exact balanced
        candidate distribution used by the normal conditional head and does
        not change the existing sampling or PPO evaluation methods.
        """

        if operation_slice is None:
            raise ValueError(
                "grouped deterministic decoding requires operation groups"
            )
        scores, single = self._candidate_scores(
            observation,
            candidate_features,
            feature_dim,
            scorer,
        )
        if not single or scores.ndim != 2 or int(scores.shape[0]) != 1:
            raise ValueError("conditional candidate sampling 需要单个 context。")
        candidate_features_t = torch.as_tensor(
            candidate_features,
            dtype=torch.float32,
            device=scores.device,
        )
        distribution_logits = _group_balance_candidate_logits(
            scores.squeeze(0),
            candidate_features_t,
            None,
            operation_slice,
        )
        distribution = torch.distributions.Categorical(logits=distribution_logits)
        start = 0 if operation_slice.start is None else int(operation_slice.start)
        stop = (
            int(candidate_features_t.shape[-1])
            if operation_slice.stop is None
            else int(operation_slice.stop)
        )
        operation_values = candidate_features_t[:, start:stop]
        if operation_values.ndim != 2 or int(operation_values.shape[0]) != int(
            distribution.probs.shape[-1]
        ):
            raise ValueError("candidate operation features shape mismatch")
        operation_indices = torch.argmax(operation_values, dim=-1)
        group_count = stop - start
        group_probabilities = torch.stack(
            [
                distribution.probs[operation_indices == group_index].sum()
                for group_index in range(group_count)
            ]
        )
        selected_group = int(torch.argmax(group_probabilities).item())
        group_candidates = torch.nonzero(
            operation_indices == selected_group,
            as_tuple=False,
        ).reshape(-1)
        if int(group_candidates.numel()) == 0:
            raise ValueError("grouped deterministic decoding selected an empty group")
        selected = int(
            group_candidates[
                int(torch.argmax(distribution.probs[group_candidates]).item())
            ].item()
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
        return CandidatePolicySample(
            selected_index=selected,
            log_prob=distribution.log_prob(selected_tensor),
            entropy=distribution.entropy(),
            logits=distribution_logits,
        )

    def sample_cache_candidate(
        self,
        observation: Any,
        candidate_features: Any,
        deterministic: bool = False,
    ) -> CandidatePolicySample:
        return self._sample_candidate(
            observation,
            candidate_features,
            self.cache_feature_dim,
            self.cache_candidate_scorer,
            deterministic,
            self._candidate_operation_slice("cache"),
        )

    def sample_uav_candidate(
        self,
        observation: Any,
        candidate_features: Any,
        deterministic: bool = False,
    ) -> CandidatePolicySample:
        return self._sample_candidate(
            observation,
            candidate_features,
            self.uav_feature_dim,
            self.uav_candidate_scorer,
            deterministic,
            self._candidate_operation_slice("uav"),
        )

    def sample_cache_candidate_grouped_deterministic(
        self,
        observation: Any,
        candidate_features: Any,
    ) -> CandidatePolicySample:
        return self._sample_candidate_grouped_deterministic(
            observation,
            candidate_features,
            self.cache_feature_dim,
            self.cache_candidate_scorer,
            self._candidate_operation_slice("cache"),
        )

    def sample_uav_candidate_grouped_deterministic(
        self,
        observation: Any,
        candidate_features: Any,
    ) -> CandidatePolicySample:
        return self._sample_candidate_grouped_deterministic(
            observation,
            candidate_features,
            self.uav_feature_dim,
            self.uav_candidate_scorer,
            self._candidate_operation_slice("uav"),
        )

    def _evaluate_conditional_heads(
        self,
        contexts: Any,
        candidate_features: Any,
        candidate_masks: Any,
        selected_indices: Any,
        active_heads: Any,
        feature_dim: int,
        scorer: Any,
        operation_slice: slice | None = None,
    ) -> tuple[Any, Any]:
        contexts_t = torch.as_tensor(contexts, dtype=torch.float32)
        features_t = torch.as_tensor(candidate_features, dtype=torch.float32)
        masks_t = torch.as_tensor(candidate_masks, dtype=torch.bool)
        selected_t = torch.as_tensor(selected_indices, dtype=torch.long)
        active_t = torch.as_tensor(active_heads, dtype=torch.bool)
        if contexts_t.ndim != 3:
            raise ValueError("conditional contexts 必须为 [B,H,observation_dim]。")
        batch_size, head_count, observation_dim = [int(value) for value in contexts_t.shape]
        if observation_dim != self.observation_dim:
            raise ValueError("conditional context observation_dim 不匹配。")
        if tuple(features_t.shape[:2]) != (batch_size, head_count) or int(
            features_t.shape[-1]
        ) != int(feature_dim):
            raise ValueError("conditional candidate feature shape 不匹配。")
        candidate_count = int(features_t.shape[2])
        if tuple(masks_t.shape) != (batch_size, head_count, candidate_count):
            raise ValueError("conditional candidate mask shape 不匹配。")
        if tuple(selected_t.shape) != (batch_size, head_count) or tuple(active_t.shape) != (
            batch_size,
            head_count,
        ):
            raise ValueError("conditional selected/active shape 不匹配。")
        flat_scores, _ = self._candidate_scores(
            contexts_t.reshape(batch_size * head_count, observation_dim),
            features_t.reshape(batch_size * head_count, candidate_count, feature_dim),
            feature_dim,
            scorer,
        )
        scores = flat_scores.reshape(batch_size, head_count, candidate_count)
        if not bool(torch.all(torch.any(masks_t, dim=-1)).item()):
            raise ValueError("每个 conditional head 至少需要一个 padded valid candidate。")
        if bool(torch.any(selected_t < 0).item()) or bool(
            torch.any(selected_t >= candidate_count).item()
        ):
            raise ValueError("conditional selected index 越界。")
        selected_valid = masks_t.gather(-1, selected_t.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.all(selected_valid).item()):
            raise ValueError("conditional selected index 被 candidate mask 屏蔽。")
        distribution_logits = scores.masked_fill(~masks_t.to(scores.device), -float("inf"))
        if operation_slice is not None:
            distribution_logits = _group_balance_candidate_logits(
                scores,
                features_t,
                masks_t,
                operation_slice,
                active_heads=active_t,
            )
        distribution = torch.distributions.Categorical(logits=distribution_logits)
        active_float = active_t.to(torch.float32)
        log_prob = (distribution.log_prob(selected_t) * active_float).sum(dim=-1)
        entropy = (distribution.entropy() * active_float).sum(dim=-1)
        return log_prob, entropy

    def evaluate_primitive(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any,
        cache_contexts: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_selected_indices: Any,
        uav_contexts: Any,
        uav_candidate_features: Any,
        uav_candidate_masks: Any,
        uav_selected_indices: Any,
        num_cache_heads: int,
    ) -> PolicyEvaluation:
        trigger = super().evaluate(observations, actions, action_masks)
        action_values = _validate_binary_actions(actions, self.action_dim).to(
            trigger.log_prob.device
        )
        cache_active = action_values[:, : int(num_cache_heads)].to(torch.bool)
        uav_active = action_values[:, int(num_cache_heads) :].to(torch.bool)
        cache_log_prob, cache_entropy = self._evaluate_conditional_heads(
            cache_contexts,
            cache_candidate_features,
            cache_candidate_masks,
            cache_selected_indices,
            cache_active,
            self.cache_feature_dim,
            self.cache_candidate_scorer,
            self._candidate_operation_slice("cache"),
        )
        uav_log_prob, uav_entropy = self._evaluate_conditional_heads(
            uav_contexts,
            uav_candidate_features,
            uav_candidate_masks,
            uav_selected_indices,
            uav_active,
            self.uav_feature_dim,
            self.uav_candidate_scorer,
            self._candidate_operation_slice("uav"),
        )
        return PolicyEvaluation(
            log_prob=trigger.log_prob + cache_log_prob + uav_log_prob,
            entropy=trigger.entropy + cache_entropy + uav_entropy,
            value=trigger.value,
            logits=trigger.logits,
            cache_log_prob=cache_log_prob,
            uav_log_prob=uav_log_prob,
            cache_entropy=cache_entropy,
            uav_entropy=uav_entropy,
        )


# The formal V6 network deliberately inherits through the following schema
# generations.  This chain fixes its module hierarchy and state_dict keys; the
# intermediate classes are implementation bases, not active experiment choices.
class AdaptivePrimitiveV2ActorCritic(AdaptivePrimitiveActorCritic):
    """Scheme B policy with the v2 finite-horizon observation schema."""

    architecture_id = "adaptive_primitive_v2"


class AdaptivePrimitiveGroupedV2ActorCritic(AdaptivePrimitiveV2ActorCritic):
    """v2 policy with operation-group-balanced conditional candidate logits."""

    architecture_id = "adaptive_primitive_v2_grouped"
    architecture_version = 1

    def _candidate_operation_slice(self, candidate_kind: str) -> slice:
        if candidate_kind == "cache":
            return slice(4, 8)
        if candidate_kind == "uav":
            uav_count = int(self.uav_feature_dim) - 16
            if uav_count <= 0:
                raise ValueError("UAV candidate feature dimension 缺少 UAV identity one-hot。")
            return slice(1 + uav_count, 3 + uav_count)
        raise ValueError(f"unknown candidate kind={candidate_kind}")


class AdaptivePrimitiveHorizonGroupedV3ActorCritic(AdaptivePrimitiveGroupedV2ActorCritic):
    """Grouped primitive policy using the explicit full prediction-horizon state."""

    architecture_id = "adaptive_primitive_v3_horizon_grouped"
    architecture_version = 1


class AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic(
    AdaptivePrimitiveHorizonGroupedV3ActorCritic
):
    """V3 horizon/grouped policy with an independent critic representation."""

    architecture_id = "adaptive_primitive_v4_horizon_grouped_separate_critic"
    architecture_version = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            cache_feature_dim,
            uav_feature_dim,
            hidden_sizes,
        )
        critic_layers: list[Any] = []
        last_dim = self.observation_dim
        for hidden_size in hidden_sizes:
            critic_layers.extend([nn.Linear(last_dim, int(hidden_size)), nn.Tanh()])
            last_dim = int(hidden_size)
        self.critic_encoder = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(last_dim, 1)

    def _critic_values(self, observations: Any, actor_hidden: Any) -> Any:
        del actor_hidden
        return self.critic_head(self.critic_encoder(observations)).squeeze(-1)


class AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic(
    AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic
):
    """V4 network with explicit disjoint actor/critic parameter groups for V5."""

    architecture_id = "adaptive_primitive_v5_horizon_grouped_separate_critic_optimization"
    architecture_version = 1

    def actor_parameters(self) -> tuple[Any, ...]:
        """Return only parameters updated by the actor optimizer."""

        modules = (
            self.encoder,
            self.actor_head,
            self.cache_candidate_scorer,
            self.uav_candidate_scorer,
        )
        parameters = tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("V5 actor parameter group contains duplicate parameters.")
        return parameters

    def critic_parameters(self) -> tuple[Any, ...]:
        """Return only parameters updated by the critic optimizer."""

        modules = (self.critic_encoder, self.critic_head)
        parameters = tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("V5 critic parameter group contains duplicate parameters.")
        return parameters

    def critic_value(self, observations: Any) -> Any:
        """Evaluate the independent critic without building an actor graph."""

        obs, single = self._observations(observations)
        values = self.critic_head(self.critic_encoder(obs)).squeeze(-1)
        return values.squeeze(0) if single else values


class AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic(
    AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic
):
    """V5 policy whose independent critic predicts a PopArt-normalized value."""

    architecture_id = "adaptive_primitive_v6_horizon_grouped_separate_critic_popart"
    architecture_version = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            cache_feature_dim,
            uav_feature_dim,
            hidden_sizes,
        )
        self.register_buffer("popart_mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("popart_std", torch.ones((), dtype=torch.float32))

    @staticmethod
    def _validate_popart_statistics(mean: Any, std: Any) -> tuple[float, float]:
        mean_f = float(mean)
        std_f = float(std)
        if not math.isfinite(mean_f) or not math.isfinite(std_f) or std_f <= 0.0:
            raise ValueError("PopArt mean/std must be finite and std must be positive.")
        return mean_f, std_f

    def popart_statistics(self) -> tuple[float, float]:
        """Return the active raw-value mean and standard deviation."""

        return (
            float(self.popart_mean.detach().cpu().item()),
            float(self.popart_std.detach().cpu().item()),
        )

    def critic_normalized_value(self, observations: Any) -> Any:
        """Return the critic's internal normalized value z(s)."""

        obs, single = self._observations(observations)
        values = self.critic_head(self.critic_encoder(obs)).squeeze(-1)
        return values.squeeze(0) if single else values

    def critic_value(self, observations: Any) -> Any:
        """Return the raw value V(s) exposed to rollout, GAE, and PPO."""

        normalized_values = self.critic_normalized_value(observations)
        return self.popart_mean + self.popart_std * normalized_values

    def _critic_values(self, observations: Any, actor_hidden: Any) -> Any:
        del actor_hidden
        return self.critic_value(observations)

    def update_popart_statistics(self, mean: Any, std: Any) -> None:
        """Transform only the critic output head while preserving raw values."""

        mean_new, std_new = self._validate_popart_statistics(mean, std)
        mean_old, std_old = self.popart_statistics()
        with torch.no_grad():
            self.critic_head.weight.mul_(std_old / std_new)
            self.critic_head.bias.mul_(std_old).add_(mean_old - mean_new).div_(std_new)
            self.popart_mean.fill_(mean_new)
            self.popart_std.fill_(std_new)


class AdaptiveCacheOperationGroupV7PopArtActorCritic(
    AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic
):
    """V7 Cache-only hierarchical policy on top of the frozen V6 backbone.

    The inherited V6 binary trigger head and UAV scorer remain in the module
    hierarchy for checkpoint compatibility, but V7 training does not use
    either of them.  Cache operation logits are produced by a new per-node
    four-way head, while candidate logits are the raw cache scorer scores
    restricted to the operation selected by that head.
    """

    architecture_id = "adaptive_cache_operation_group_v7_popart"
    architecture_version = 1
    operation_order = ("NO_CHANGE", "ADD", "DELETE", "SWAP")

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        num_cache_heads: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            cache_feature_dim,
            uav_feature_dim,
            hidden_sizes,
        )
        if int(num_cache_heads) <= 0:
            raise ValueError("V7 num_cache_heads must be positive")
        self.num_cache_heads = int(num_cache_heads)
        self.cache_operation_head = nn.Linear(
            self.hidden_dim,
            self.num_cache_heads * len(self.operation_order),
        )

    def actor_parameters(self) -> tuple[Any, ...]:
        """Return only parameters used by the V7 Cache policy."""

        modules = (
            self.encoder,
            self.cache_candidate_scorer,
            self.cache_operation_head,
        )
        parameters = tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("V7 actor parameter group contains duplicate parameters.")
        return parameters

    def _operation_logits(self, observations: Any) -> tuple[Any, bool]:
        obs, single = self._observations(observations)
        hidden = self.encoder(obs)
        logits = self.cache_operation_head(hidden).reshape(
            int(obs.shape[0]),
            self.num_cache_heads,
            len(self.operation_order),
        )
        return (logits.squeeze(0) if single else logits), single

    def sample_cache_operation(
        self,
        observation: Any,
        head_index: int,
        operation_mask: Any,
        deterministic: bool = False,
    ) -> OperationPolicySample:
        """Sample one node's operation using its dynamic feasibility mask."""

        logits, single = self._operation_logits(observation)
        if not single or tuple(logits.shape) != (
            self.num_cache_heads,
            len(self.operation_order),
        ):
            raise ValueError("V7 operation sampling requires one observation")
        index = int(head_index)
        if index < 0 or index >= self.num_cache_heads:
            raise IndexError("V7 cache operation head index out of range")
        mask = torch.as_tensor(
            np.asarray(operation_mask, dtype=bool).reshape(-1),
            dtype=torch.bool,
            device=logits.device,
        )
        if tuple(mask.shape) != (len(self.operation_order),) or not bool(
            torch.any(mask).item()
        ):
            raise ValueError("V7 operation mask must allow at least one operation")
        raw_logits = logits[index]
        raw_distribution = torch.distributions.Categorical(logits=raw_logits)
        masked_logits = raw_logits.masked_fill(~mask, -float("inf"))
        distribution = torch.distributions.Categorical(logits=masked_logits)
        selected = (
            int(torch.argmax(distribution.logits).item())
            if deterministic
            else int(distribution.sample().item())
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
        return OperationPolicySample(
            selected_index=selected,
            log_prob=distribution.log_prob(selected_tensor),
            entropy=distribution.entropy(),
            logits=masked_logits,
            raw_logits=raw_logits,
            probabilities=raw_distribution.probs,
        )

    def sample_cache_candidate_v7(
        self,
        observation: Any,
        candidate_features: Any,
        candidate_mask: Any | None = None,
        deterministic: bool = False,
    ) -> CandidatePolicySample:
        """Sample only within one operation group using raw scorer scores."""

        scores, single = self._candidate_scores(
            observation,
            candidate_features,
            self.cache_feature_dim,
            self.cache_candidate_scorer,
        )
        if not single or scores.ndim != 2 or int(scores.shape[0]) != 1:
            raise ValueError("V7 candidate sampling requires one context")
        raw_scores = scores.squeeze(0)
        if candidate_mask is None:
            mask = torch.ones_like(raw_scores, dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                np.asarray(candidate_mask, dtype=bool).reshape(-1),
                dtype=torch.bool,
                device=raw_scores.device,
            )
        if tuple(mask.shape) != tuple(raw_scores.shape) or not bool(
            torch.any(mask).item()
        ):
            raise ValueError("V7 candidate mask must allow at least one candidate")
        distribution_logits = raw_scores.masked_fill(~mask, -float("inf"))
        distribution = torch.distributions.Categorical(logits=distribution_logits)
        selected = (
            int(torch.argmax(distribution.logits).item())
            if deterministic
            else int(distribution.sample().item())
        )
        selected_tensor = torch.as_tensor(
            selected,
            dtype=torch.long,
            device=raw_scores.device,
        )
        return CandidatePolicySample(
            selected_index=selected,
            log_prob=distribution.log_prob(selected_tensor),
            entropy=distribution.entropy(),
            logits=distribution_logits,
        )

    def evaluate_cache_operation_primitive(
        self,
        observations: Any,
        operation_contexts: Any,
        operation_masks: Any,
        operation_actions: Any,
        candidate_features: Any,
        candidate_masks: Any,
        candidate_selected_indices: Any,
    ) -> PolicyEvaluation:
        """Evaluate the V7 joint operation-plus-candidate distribution."""

        observations_t, _ = self._observations(observations)
        operation_contexts_t = torch.as_tensor(
            operation_contexts,
            dtype=torch.float32,
            device=observations_t.device,
        )
        operation_masks_t = torch.as_tensor(
            operation_masks,
            dtype=torch.bool,
            device=observations_t.device,
        )
        operation_actions_t = torch.as_tensor(
            operation_actions,
            dtype=torch.long,
            device=observations_t.device,
        )
        candidate_features_t = torch.as_tensor(
            candidate_features,
            dtype=torch.float32,
            device=observations_t.device,
        )
        candidate_masks_t = torch.as_tensor(
            candidate_masks,
            dtype=torch.bool,
            device=observations_t.device,
        )
        candidate_selected_t = torch.as_tensor(
            candidate_selected_indices,
            dtype=torch.long,
            device=observations_t.device,
        )
        if operation_contexts_t.ndim != 3:
            raise ValueError("V7 operation_contexts must be [B,H,observation_dim]")
        batch_size, head_count, observation_dim = [
            int(value) for value in operation_contexts_t.shape
        ]
        if batch_size != int(observations_t.shape[0]):
            raise ValueError("V7 operation_contexts batch dimension mismatch")
        if head_count != self.num_cache_heads or observation_dim != self.observation_dim:
            raise ValueError("V7 operation_contexts shape mismatch")
        expected_operation_shape = (
            batch_size,
            self.num_cache_heads,
            len(self.operation_order),
        )
        if tuple(operation_masks_t.shape) != expected_operation_shape:
            raise ValueError("V7 operation_masks shape mismatch")
        if tuple(operation_actions_t.shape) != (
            batch_size,
            self.num_cache_heads,
        ):
            raise ValueError("V7 operation_actions shape mismatch")
        if bool(torch.any(operation_actions_t < 0).item()) or bool(
            torch.any(operation_actions_t >= len(self.operation_order)).item()
        ):
            raise ValueError("V7 operation action index out of range")
        if not bool(torch.all(torch.any(operation_masks_t, dim=-1)).item()):
            raise ValueError("V7 operation mask cannot block every operation")

        all_operation_logits, _ = self._operation_logits(
            operation_contexts_t.reshape(batch_size * head_count, observation_dim)
        )
        all_operation_logits = all_operation_logits.reshape(
            batch_size,
            head_count,
            self.num_cache_heads,
            len(self.operation_order),
        )
        head_indices = torch.arange(
            self.num_cache_heads,
            dtype=torch.long,
            device=all_operation_logits.device,
        )
        operation_logits = all_operation_logits[:, head_indices, head_indices, :]
        if tuple(operation_logits.shape) != expected_operation_shape:
            raise AssertionError("V7 operation head diagonal shape mismatch")
        operation_logits = operation_logits.masked_fill(
            ~operation_masks_t,
            -float("inf"),
        )
        operation_distribution = torch.distributions.Categorical(logits=operation_logits)
        operation_valid = operation_masks_t.gather(
            -1,
            operation_actions_t.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.all(operation_valid).item()):
            raise ValueError("V7 operation action is blocked by its operation mask")
        operation_node_log_prob = operation_distribution.log_prob(operation_actions_t)
        operation_log_prob = operation_node_log_prob.sum(dim=-1)
        operation_entropy = torch.where(
            operation_masks_t.sum(dim=-1) == 1,
            torch.zeros_like(operation_distribution.entropy()),
            operation_distribution.entropy(),
        ).sum(dim=-1)

        if candidate_features_t.ndim != 4:
            raise ValueError("V7 candidate_features must be [B,H,C,F]")
        if tuple(candidate_features_t.shape[:2]) != (
            batch_size,
            self.num_cache_heads,
        ) or int(candidate_features_t.shape[-1]) != self.cache_feature_dim:
            raise ValueError("V7 candidate_features shape mismatch")
        candidate_count = int(candidate_features_t.shape[2])
        if candidate_count <= 0:
            raise ValueError("V7 candidate tensor must have positive width")
        expected_candidate_shape = (
            batch_size,
            self.num_cache_heads,
            candidate_count,
        )
        if tuple(candidate_masks_t.shape) != expected_candidate_shape:
            raise ValueError("V7 candidate_masks shape mismatch")
        if tuple(candidate_selected_t.shape) != (
            batch_size,
            self.num_cache_heads,
        ):
            raise ValueError("V7 candidate_selected_indices shape mismatch")
        if not bool(torch.all(torch.any(candidate_masks_t, dim=-1)).item()):
            raise ValueError("V7 candidate mask must have a placeholder or active candidate")
        if bool(torch.any(candidate_selected_t < 0).item()) or bool(
            torch.any(candidate_selected_t >= candidate_count).item()
        ):
            raise ValueError("V7 candidate selected index out of range")
        candidate_valid = candidate_masks_t.gather(
            -1,
            candidate_selected_t.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.all(candidate_valid).item()):
            raise ValueError("V7 candidate selected index is masked")
        candidate_contexts = operation_contexts_t.reshape(
            batch_size * head_count,
            observation_dim,
        )
        flat_features = candidate_features_t.reshape(
            batch_size * head_count,
            candidate_count,
            self.cache_feature_dim,
        )
        raw_scores, _ = self._candidate_scores(
            candidate_contexts,
            flat_features,
            self.cache_feature_dim,
            self.cache_candidate_scorer,
        )
        raw_scores = raw_scores.reshape(
            batch_size,
            self.num_cache_heads,
            candidate_count,
        )
        candidate_logits = raw_scores.masked_fill(~candidate_masks_t, -float("inf"))
        candidate_distribution = torch.distributions.Categorical(logits=candidate_logits)
        candidate_active = operation_actions_t != 0
        candidate_node_log_prob = (
            candidate_distribution.log_prob(candidate_selected_t)
            * candidate_active.to(torch.float32)
        )
        candidate_log_prob = candidate_node_log_prob.sum(dim=-1)
        candidate_entropy = (
            candidate_distribution.entropy()
            * candidate_active.to(torch.float32)
        ).sum(dim=-1)
        return PolicyEvaluation(
            log_prob=operation_log_prob + candidate_log_prob,
            entropy=operation_entropy + candidate_entropy,
            value=self.critic_value(observations_t),
            logits=operation_logits,
            cache_log_prob=operation_log_prob + candidate_log_prob,
            cache_entropy=operation_entropy + candidate_entropy,
            cache_node_log_probs=operation_node_log_prob + candidate_node_log_prob,
        )

class AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic(
    AdaptiveCacheOperationGroupV7PopArtActorCritic
):
    """V7 Cache operation hierarchy with six replica-aware candidate features."""

    architecture_id = "adaptive_cache_operation_group_replica_aware_v8_popart"
    architecture_version = 1
    replica_aware_candidate_features = True


class AdaptiveJointCacheReplicaAwareUAVOperationJointActorCritic(
    AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic
):
    """Neutral joint Cache/UAV policy base with explicit UAV operations."""

    architecture_id = "adaptive_joint_cache_replica_aware_uav_operation_joint_base"
    architecture_version = 1
    replica_aware_candidate_features = True
    uav_operation_order = ("STAY", "MOVE")

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        num_cache_heads: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            cache_feature_dim,
            uav_feature_dim,
            num_cache_heads,
            hidden_sizes,
        )
        if int(uav_feature_dim) <= 16:
            raise ValueError("uav_feature_dim must include at least one UAV head")
        self.num_uav_heads = int(uav_feature_dim) - 16
        self.uav_operation_head = nn.Linear(
            self.hidden_dim,
            self.num_uav_heads * len(self.uav_operation_order),
        )

    def actor_parameters(self) -> tuple[Any, ...]:
        """Return exactly the five modules used by the joint actor."""

        modules = (
            self.encoder,
            self.cache_operation_head,
            self.cache_candidate_scorer,
            self.uav_operation_head,
            self.uav_candidate_scorer,
        )
        parameters = tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("actor parameter group contains duplicate parameters.")
        return parameters

    def actor_gradient_parameter_groups(self) -> dict[str, tuple[Any, ...]]:
        """Return disjoint actor groups for post-backward diagnostics only.

        The actor still has one optimizer and one joint PPO loss.  The shared
        encoder is intentionally kept separate from the branch-only modules so
        its gradient is not attributed to either branch.
        """

        groups = {
            "cache": tuple(
                parameter
                for module in (
                    self.cache_operation_head,
                    self.cache_candidate_scorer,
                )
                for parameter in module.parameters()
            ),
            "uav": tuple(
                parameter
                for module in (
                    self.uav_operation_head,
                    self.uav_candidate_scorer,
                )
                for parameter in module.parameters()
            ),
            "shared": tuple(self.encoder.parameters()),
        }
        group_ids = [
            id(parameter)
            for parameters in groups.values()
            for parameter in parameters
        ]
        if len(set(group_ids)) != len(group_ids):
            raise RuntimeError("actor gradient groups overlap.")
        actor_ids = {id(parameter) for parameter in self.actor_parameters()}
        if set(group_ids) != actor_ids:
            raise RuntimeError("actor gradient groups do not cover the actor.")
        return groups

    def _uav_operation_logits(self, observations: Any) -> tuple[Any, bool]:
        obs, single = self._observations(observations)
        hidden = self.encoder(obs)
        logits = self.uav_operation_head(hidden).reshape(
            int(obs.shape[0]),
            self.num_uav_heads,
            len(self.uav_operation_order),
        )
        return (logits.squeeze(0) if single else logits), single

    def sample_uav_operation(
        self,
        observation: Any,
        head_index: int,
        operation_mask: Any,
        deterministic: bool = False,
    ) -> OperationPolicySample:
        """Sample one UAV's explicit STAY/MOVE operation."""

        logits, single = self._uav_operation_logits(observation)
        if not single or tuple(logits.shape) != (
            self.num_uav_heads,
            len(self.uav_operation_order),
        ):
            raise ValueError("UAV operation sampling requires one observation")
        index = int(head_index)
        if index < 0 or index >= self.num_uav_heads:
            raise IndexError("UAV operation head index out of range")
        raw_operation_logits = logits[index]
        raw_operation_probabilities = torch.softmax(raw_operation_logits, dim=-1)
        mask = torch.as_tensor(
            np.asarray(operation_mask, dtype=bool).reshape(-1),
            dtype=torch.bool,
            device=logits.device,
        )
        if tuple(mask.shape) != (len(self.uav_operation_order),) or not bool(
            torch.any(mask).item()
        ):
            raise ValueError("UAV operation mask must allow at least one operation")
        masked_logits = logits[index].masked_fill(~mask, -float("inf"))
        distribution = torch.distributions.Categorical(logits=masked_logits)
        selected = (
            int(torch.argmax(distribution.logits).item())
            if deterministic
            else int(distribution.sample().item())
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
        return OperationPolicySample(
            selected_index=selected,
            log_prob=distribution.log_prob(selected_tensor),
            entropy=distribution.entropy(),
            logits=masked_logits,
            raw_logits=raw_operation_logits.detach(),
            probabilities=raw_operation_probabilities.detach(),
        )

    def evaluate_joint_cache_uav_operation(
        self,
        observations: Any,
        cache_operation_contexts: Any,
        cache_operation_masks: Any,
        cache_operation_actions: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_candidate_selected_indices: Any,
        uav_operation_contexts: Any,
        uav_operation_masks: Any,
        uav_operation_actions: Any,
        uav_candidate_features: Any,
        uav_candidate_masks: Any,
        uav_candidate_selected_indices: Any,
    ) -> PolicyEvaluation:
        """Recompute the full joint log-probability without the old trigger head."""

        cache_evaluation = self.evaluate_cache_operation_primitive(
            observations,
            cache_operation_contexts,
            cache_operation_masks,
            cache_operation_actions,
            cache_candidate_features,
            cache_candidate_masks,
            cache_candidate_selected_indices,
        )
        observations_t, _ = self._observations(observations)
        operation_contexts_t = torch.as_tensor(
            uav_operation_contexts,
            dtype=torch.float32,
            device=observations_t.device,
        )
        operation_masks_t = torch.as_tensor(
            uav_operation_masks,
            dtype=torch.bool,
            device=observations_t.device,
        )
        operation_actions_t = torch.as_tensor(
            uav_operation_actions,
            dtype=torch.long,
            device=observations_t.device,
        )
        candidate_features_t = torch.as_tensor(
            uav_candidate_features,
            dtype=torch.float32,
            device=observations_t.device,
        )
        candidate_masks_t = torch.as_tensor(
            uav_candidate_masks,
            dtype=torch.bool,
            device=observations_t.device,
        )
        candidate_selected_t = torch.as_tensor(
            uav_candidate_selected_indices,
            dtype=torch.long,
            device=observations_t.device,
        )
        if operation_contexts_t.ndim != 3:
            raise ValueError("UAV operation_contexts must be [B,U,observation_dim]")
        batch_size, head_count, observation_dim = [
            int(value) for value in operation_contexts_t.shape
        ]
        if batch_size != int(observations_t.shape[0]):
            raise ValueError("UAV operation_contexts batch dimension mismatch")
        if head_count != self.num_uav_heads or observation_dim != self.observation_dim:
            raise ValueError("UAV operation_contexts shape mismatch")
        expected_operation_shape = (
            batch_size,
            self.num_uav_heads,
            len(self.uav_operation_order),
        )
        if tuple(operation_masks_t.shape) != expected_operation_shape:
            raise ValueError("UAV operation_masks shape mismatch")
        if tuple(operation_actions_t.shape) != (batch_size, self.num_uav_heads):
            raise ValueError("UAV operation_actions shape mismatch")
        if bool(torch.any(operation_actions_t < 0).item()) or bool(
            torch.any(operation_actions_t >= len(self.uav_operation_order)).item()
        ):
            raise ValueError("UAV operation action index out of range")
        if not bool(torch.all(torch.any(operation_masks_t, dim=-1)).item()):
            raise ValueError("UAV operation mask cannot block every operation")
        all_operation_logits, _ = self._uav_operation_logits(
            operation_contexts_t.reshape(batch_size * head_count, observation_dim)
        )
        all_operation_logits = all_operation_logits.reshape(
            batch_size,
            head_count,
            self.num_uav_heads,
            len(self.uav_operation_order),
        )
        head_indices = torch.arange(
            self.num_uav_heads,
            dtype=torch.long,
            device=all_operation_logits.device,
        )
        operation_logits = all_operation_logits[:, head_indices, head_indices, :]
        operation_logits = operation_logits.masked_fill(~operation_masks_t, -float("inf"))
        operation_distribution = torch.distributions.Categorical(logits=operation_logits)
        operation_valid = operation_masks_t.gather(
            -1,
            operation_actions_t.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.all(operation_valid).item()):
            raise ValueError("UAV operation action is blocked by its operation mask")
        uav_operation_log_probs = operation_distribution.log_prob(operation_actions_t)
        operation_log_prob = uav_operation_log_probs.sum(dim=-1)
        operation_entropy = torch.where(
            operation_masks_t.sum(dim=-1) == 1,
            torch.zeros_like(operation_distribution.entropy()),
            operation_distribution.entropy(),
        ).sum(dim=-1)

        if candidate_features_t.ndim != 4:
            raise ValueError("UAV candidate_features must be [B,U,C,F]")
        if tuple(candidate_features_t.shape[:2]) != (
            batch_size,
            self.num_uav_heads,
        ) or int(candidate_features_t.shape[-1]) != self.uav_feature_dim:
            raise ValueError("UAV candidate_features shape mismatch")
        candidate_count = int(candidate_features_t.shape[2])
        if candidate_count <= 0:
            raise ValueError("UAV candidate tensor must have positive width")
        expected_candidate_shape = (batch_size, self.num_uav_heads, candidate_count)
        if tuple(candidate_masks_t.shape) != expected_candidate_shape:
            raise ValueError("UAV candidate_masks shape mismatch")
        if tuple(candidate_selected_t.shape) != (batch_size, self.num_uav_heads):
            raise ValueError("UAV candidate_selected_indices shape mismatch")
        if not bool(torch.all(torch.any(candidate_masks_t, dim=-1)).item()):
            raise ValueError("UAV candidate mask must have a placeholder or candidate")
        if bool(torch.any(candidate_selected_t < 0).item()) or bool(
            torch.any(candidate_selected_t >= candidate_count).item()
        ):
            raise ValueError("UAV candidate selected index out of range")
        candidate_valid = candidate_masks_t.gather(
            -1,
            candidate_selected_t.unsqueeze(-1),
        ).squeeze(-1)
        if not bool(torch.all(candidate_valid).item()):
            raise ValueError("UAV candidate selected index is masked")
        candidate_active = operation_actions_t == 1
        move_feature_slice = slice(1 + self.num_uav_heads, 3 + self.num_uav_heads)
        selected_operation_features = candidate_features_t[
            ..., move_feature_slice
        ].gather(
            -2,
            candidate_selected_t.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, 2
            ),
        ).squeeze(-2)
        if bool(
            torch.any(
                candidate_active
                & ~torch.all(
                    selected_operation_features
                    == torch.as_tensor(
                        [0.0, 1.0],
                        dtype=torch.float32,
                        device=observations_t.device,
                    ),
                    dim=-1,
                )
            ).item()
        ):
            raise ValueError("MOVE operation requires a MOVE candidate")
        candidate_contexts = operation_contexts_t.reshape(
            batch_size * head_count,
            observation_dim,
        )
        flat_features = candidate_features_t.reshape(
            batch_size * head_count,
            candidate_count,
            self.uav_feature_dim,
        )
        raw_scores, _ = self._candidate_scores(
            candidate_contexts,
            flat_features,
            self.uav_feature_dim,
            self.uav_candidate_scorer,
        )
        raw_scores = raw_scores.reshape(
            batch_size,
            self.num_uav_heads,
            candidate_count,
        )
        candidate_logits = raw_scores.masked_fill(~candidate_masks_t, -float("inf"))
        candidate_distribution = torch.distributions.Categorical(logits=candidate_logits)
        uav_candidate_log_probs = (
            candidate_distribution.log_prob(candidate_selected_t)
            * candidate_active.to(torch.float32)
        )
        candidate_log_prob = uav_candidate_log_probs.sum(dim=-1)
        candidate_entropy = (
            candidate_distribution.entropy()
            * candidate_active.to(torch.float32)
        ).sum(dim=-1)
        return PolicyEvaluation(
            log_prob=cache_evaluation.log_prob + operation_log_prob + candidate_log_prob,
            entropy=cache_evaluation.entropy + operation_entropy + candidate_entropy,
            value=cache_evaluation.value,
            logits=cache_evaluation.logits,
            cache_log_prob=cache_evaluation.log_prob,
            uav_log_prob=operation_log_prob + candidate_log_prob,
            uav_operation_log_probs=uav_operation_log_probs,
            uav_candidate_log_probs=uav_candidate_log_probs,
            uav_candidate_log_probs_all=candidate_distribution.logits,
            cache_entropy=cache_evaluation.entropy,
            uav_entropy=operation_entropy + candidate_entropy,
            cache_node_log_probs=cache_evaluation.cache_node_log_probs,
        )


class AdaptiveJointCacheReplicaAwareUAVOperationDecoupledEncoderActorCritic(
    AdaptiveJointCacheReplicaAwareUAVOperationJointActorCritic
):
    """Neutral decoupled-encoder joint Cache/UAV policy base.

    The joint base constructor creates the shared encoder and all existing
    heads in the historical initialization order.  The encoder is then cloned
    into two independent modules without consuming additional random numbers;
    the critic and all head/scorer initialization therefore remain unchanged.
    """

    architecture_id = (
        "adaptive_joint_cache_replica_aware_uav_operation_decoupled_encoder"
    )
    architecture_version = 1

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        cache_feature_dim: int,
        uav_feature_dim: int,
        num_cache_heads: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            cache_feature_dim,
            uav_feature_dim,
            num_cache_heads,
            hidden_sizes,
        )
        original_encoder = self.encoder
        self.cache_encoder = deepcopy(original_encoder)
        self.uav_encoder = deepcopy(original_encoder)
        del self.encoder

    def _default_actor_encoder(self) -> Any:
        return self.cache_encoder

    def _candidate_encoder_for_scorer(self, scorer: Any) -> Any:
        if scorer is self.cache_candidate_scorer:
            return self.cache_encoder
        if scorer is self.uav_candidate_scorer:
            return self.uav_encoder
        raise ValueError("candidate scorer has no registered branch encoder")

    def _operation_logits(self, observations: Any) -> tuple[Any, bool]:
        obs, single = self._observations(observations)
        hidden = self.cache_encoder(obs)
        logits = self.cache_operation_head(hidden).reshape(
            int(obs.shape[0]),
            self.num_cache_heads,
            len(self.operation_order),
        )
        return (logits.squeeze(0) if single else logits), single

    def _uav_operation_logits(self, observations: Any) -> tuple[Any, bool]:
        obs, single = self._observations(observations)
        hidden = self.uav_encoder(obs)
        logits = self.uav_operation_head(hidden).reshape(
            int(obs.shape[0]),
            self.num_uav_heads,
            len(self.uav_operation_order),
        )
        return (logits.squeeze(0) if single else logits), single

    def actor_parameters(self) -> tuple[Any, ...]:
        modules = (
            self.cache_encoder,
            self.uav_encoder,
            self.cache_operation_head,
            self.cache_candidate_scorer,
            self.uav_operation_head,
            self.uav_candidate_scorer,
        )
        parameters = tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
        )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("actor parameter group contains duplicates.")
        return parameters

    def actor_gradient_parameter_groups(self) -> dict[str, tuple[Any, ...]]:
        groups = {
            "cache": tuple(
                parameter
                for module in (
                    self.cache_operation_head,
                    self.cache_candidate_scorer,
                )
                for parameter in module.parameters()
            ),
            "uav": tuple(
                parameter
                for module in (
                    self.uav_operation_head,
                    self.uav_candidate_scorer,
                )
                for parameter in module.parameters()
            ),
            "cache_encoder": tuple(self.cache_encoder.parameters()),
            "uav_encoder": tuple(self.uav_encoder.parameters()),
            "shared": (),
        }
        group_ids = [
            id(parameter)
            for parameters in groups.values()
            for parameter in parameters
        ]
        if len(set(group_ids)) != len(group_ids):
            raise RuntimeError("actor gradient groups overlap.")
        actor_ids = {id(parameter) for parameter in self.actor_parameters()}
        if set(group_ids) != actor_ids:
            raise RuntimeError("actor gradient groups do not cover the actor.")
        return groups


class AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic(
    AdaptiveJointCacheReplicaAwareUAVOperationDecoupledEncoderActorCritic
):
    """Current Recovery R2 Cache/UAV actor-critic policy.

    Operation-credit assignment and its loss decomposition live in the trainer;
    this policy owns the frozen observation/action graph and checkpoint layout.
    """

    architecture_id = (
        "adaptive_joint_cache_replica_aware_uav_operation_recovery_r2"
    )
    architecture_version = 1


class AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR3ActorCritic(
    AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR2ActorCritic
):
    """Recovery R3 policy with the R2 architecture and fixed op-scale credit."""

    architecture_id = (
        "adaptive_joint_cache_replica_aware_uav_operation_recovery_r3"
    )
    architecture_version = 1


class AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR4ActorCritic(
    AdaptiveJointCacheReplicaAwareUAVOperationRecoveryR3ActorCritic
):
    """Recovery R4 policy with exact conditional candidate credit.

    The network and checkpoint layout remain identical to R3.  R4 changes only
    the trainer-side candidate credit and full conditional PPO expectation.
    """

    architecture_id = (
        "adaptive_joint_cache_replica_aware_uav_operation_recovery_r4"
    )
    architecture_version = 1


class AdaptiveCacheOperationGroupReplicaAwareV8NoPredictionPopArtActorCritic(
    AdaptiveCacheOperationGroupReplicaAwareV8PopArtActorCritic
):
    """V8 hierarchy with future vehicle/popularity inputs masked at rollout time."""

    architecture_id = (
        "adaptive_cache_operation_group_replica_aware_v8_no_prediction_popart"
    )
    architecture_version = 1
    replica_aware_candidate_features = True
    no_prediction_ablation = True
