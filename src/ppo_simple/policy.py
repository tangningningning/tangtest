"""PPO v2 trigger policy and adaptive primitive Cache/UAV policy schema."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: Any
    entropy: Any
    value: Any
    logits: Any


@dataclass(frozen=True)
class CandidatePolicySample:
    selected_index: int
    log_prob: Any
    entropy: Any
    logits: Any


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

    def forward(self, observations: Any) -> tuple[Any, Any]:
        obs, single = self._observations(observations)
        hidden = self.encoder(obs)
        logits = self.actor_head(hidden)
        values = self.critic_head(hidden).squeeze(-1)
        if single:
            return logits.squeeze(0), values.squeeze(0)
        return logits, values

    def _masked_distribution(self, logits: Any, masks: Any) -> Any:
        binary_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return torch.distributions.Categorical(logits=binary_logits.masked_fill(~masks, -float("inf")))

    def sample(
        self,
        observation: Any,
        action_mask: Callable[[int, np.ndarray], Any] | Any = None,
        deterministic: bool = False,
    ) -> PolicySample:
        logits, value = self.forward(observation)
        if logits.ndim != 1 or tuple(logits.shape) != (self.action_dim,):
            raise ValueError("sample 需要单个 observation。")
        actions: list[int] = []
        masks: list[np.ndarray] = []
        log_probs: list[Any] = []
        entropies: list[Any] = []
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
            value_index = (
                int(torch.argmax(distribution.logits, dim=-1).item())
                if deterministic
                else int(distribution.sample().item())
            )
            actions.append(value_index)
            masks.append(mask.copy())
            action_tensor = torch.as_tensor([value_index], dtype=torch.long, device=logits.device)
            log_probs.append(distribution.log_prob(action_tensor).squeeze(0))
            entropies.append(distribution.entropy().squeeze(0))
        return PolicySample(
            action=np.asarray(actions, dtype=np.int8),
            log_prob=torch.stack(log_probs).sum(),
            entropy=torch.stack(entropies).sum(),
            value=value,
            action_mask=np.stack(masks, axis=0),
        )

    def evaluate(
        self,
        observations: Any,
        actions: Any,
        action_masks: Any = None,
    ) -> PolicyEvaluation:
        obs, _ = self._observations(observations)
        hidden = self.encoder(obs)
        logits = self.actor_head(hidden)
        values = self.critic_head(hidden).squeeze(-1)
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
        hidden = self.encoder(obs)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, int(features.shape[1]), -1)
        scores = scorer(torch.cat([expanded_hidden, features], dim=-1)).squeeze(-1)
        return scores, single

    def _sample_candidate(
        self,
        observation: Any,
        candidate_features: Any,
        feature_dim: int,
        scorer: Any,
        deterministic: bool,
    ) -> CandidatePolicySample:
        scores, single = self._candidate_scores(
            observation,
            candidate_features,
            feature_dim,
            scorer,
        )
        if not single or scores.ndim != 2 or int(scores.shape[0]) != 1:
            raise ValueError("conditional candidate sampling 需要单个 context。")
        distribution = torch.distributions.Categorical(logits=scores.squeeze(0))
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
            logits=scores.squeeze(0),
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
        distribution = torch.distributions.Categorical(
            logits=scores.masked_fill(~masks_t, -float("inf"))
        )
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
        )
        uav_log_prob, uav_entropy = self._evaluate_conditional_heads(
            uav_contexts,
            uav_candidate_features,
            uav_candidate_masks,
            uav_selected_indices,
            uav_active,
            self.uav_feature_dim,
            self.uav_candidate_scorer,
        )
        return PolicyEvaluation(
            log_prob=trigger.log_prob + cache_log_prob + uav_log_prob,
            entropy=trigger.entropy + cache_entropy + uav_entropy,
            value=trigger.value,
            logits=trigger.logits,
        )


class AdaptivePrimitiveV2ActorCritic(AdaptivePrimitiveActorCritic):
    """Scheme B policy with the v2 finite-horizon observation schema."""

    architecture_id = "adaptive_primitive_v2"
