"""Small, policy-independent ensemble for Cache counterfactual delta-Q."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class CacheCounterfactualDeltaQSurrogate(nn.Module):
    """Predict ``Q(candidate) - Q(KEEP)`` from frozen Cache features.

    The module deliberately has no reference to the PPO policy or its
    encoders.  It is therefore safe to optimize with a separate optimizer and
    to use its output only as a screening/value-completion signal.
    """

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
