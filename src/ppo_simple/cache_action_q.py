"""Independent action-conditioned Q critic for proposed Cache credit."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class CacheActionQCritic(nn.Module):
    """Predict the action residual ``A_phi(s, a) = Q(s, a) - V(s)``.

    The module intentionally owns no PPO policy parameters, encoders, or
    critic parameters.  ``V(s)`` is supplied by the current detached global
    critic and this network only learns the action-conditioned residual.
    """

    architecture_id = "cache_action_conditioned_residual_q_mlp"
    architecture_version = 1

    def __init__(
        self,
        context_dim: int,
        node_count: int,
        candidate_feature_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        dimensions = (
            int(context_dim),
            int(node_count),
            int(candidate_feature_dim),
            int(hidden_dim),
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(
                "CacheActionQCritic dimensions must all be positive"
            )
        self.context_dim, self.node_count, self.candidate_feature_dim, self.hidden_dim = dimensions
        self.input_dim = self.context_dim + self.node_count + 1 + self.candidate_feature_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        conditional_context: Any,
        node_identity: Any,
        is_update: Any,
        candidate_feature: Any,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        context = torch.as_tensor(conditional_context, dtype=torch.float32, device=device)
        node = torch.as_tensor(node_identity, dtype=torch.float32, device=device)
        update = torch.as_tensor(is_update, dtype=torch.float32, device=device)
        feature = torch.as_tensor(candidate_feature, dtype=torch.float32, device=device)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if node.ndim == 1:
            node = node.unsqueeze(0)
        if update.ndim == 1:
            update = update.unsqueeze(-1)
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        if not (
            context.ndim == 2
            and node.ndim == 2
            and update.ndim == 2
            and feature.ndim == 2
        ):
            raise ValueError(
                "CacheActionQCritic inputs must have shapes [B,D], [B,N], [B,1], [B,F]"
            )
        batch_sizes = {
            int(context.shape[0]),
            int(node.shape[0]),
            int(update.shape[0]),
            int(feature.shape[0]),
        }
        if batch_sizes != {int(context.shape[0])}:
            raise ValueError("CacheActionQCritic input batch dimensions differ")
        if tuple(context.shape[1:]) != (self.context_dim,):
            raise ValueError("CacheActionQCritic context width mismatch")
        if tuple(node.shape[1:]) != (self.node_count,):
            raise ValueError("CacheActionQCritic node identity width mismatch")
        if tuple(update.shape[1:]) != (1,):
            raise ValueError("CacheActionQCritic is_update must have width one")
        if tuple(feature.shape[1:]) != (self.candidate_feature_dim,):
            raise ValueError("CacheActionQCritic candidate feature width mismatch")
        values = (context, node, update, feature)
        if any(not bool(torch.all(torch.isfinite(value)).item()) for value in values):
            raise ValueError("CacheActionQCritic inputs contain NaN/Inf")
        output = self.network(torch.cat(values, dim=-1))
        if not bool(torch.all(torch.isfinite(output)).item()):
            raise FloatingPointError("CacheActionQCritic output contains NaN/Inf")
        return output.squeeze(-1)
