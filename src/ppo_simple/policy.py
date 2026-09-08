"""Small independent-Bernoulli actor-critic used by PPO v2.

The policy knows only an observation vector and a binary request vector.  It
does not decide cache placement, UAV coordinates, matching, or resource
allocation; those remain environment responsibilities.
"""

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
        self.actor_head = nn.Linear(last_dim, self.action_dim)
        self.critic_head = nn.Linear(last_dim, 1)

    def _observations(self, observations: Any) -> tuple[Any, bool]:
        obs = torch.as_tensor(observations, dtype=torch.float32)
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

