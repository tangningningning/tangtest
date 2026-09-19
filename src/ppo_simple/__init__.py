"""Minimal dimension-driven PPO for Cache/UAV reconfiguration requests."""

from .policy import (
    AdaptiveCacheOnlyActorCritic,
    AdaptiveReplicaAwareCacheOnlyActorCritic,
    AdaptiveNoPredictionCacheOnlyActorCritic,
    AdaptiveJointCacheUAVPPOActorCritic,
    AdaptivePrimitiveActorCritic,
    AdaptivePrimitiveGroupedV2ActorCritic,
    AdaptivePrimitiveHorizonGroupedV3ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic,
    AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic,
    AdaptivePrimitiveV2ActorCritic,
    BernoulliActorCritic,
)

__all__ = [
    "AdaptiveCacheOnlyActorCritic",
    "AdaptiveReplicaAwareCacheOnlyActorCritic",
    "AdaptiveNoPredictionCacheOnlyActorCritic",
    "AdaptiveJointCacheUAVPPOActorCritic",
    "AdaptivePrimitiveActorCritic",
    "AdaptivePrimitiveGroupedV2ActorCritic",
    "AdaptivePrimitiveHorizonGroupedV3ActorCritic",
    "AdaptivePrimitiveHorizonGroupedSeparateCriticV4ActorCritic",
    "AdaptivePrimitiveHorizonGroupedSeparateCriticOptimizationV5ActorCritic",
    "AdaptivePrimitiveHorizonGroupedSeparateCriticPopArtV6ActorCritic",
    "AdaptivePrimitiveV2ActorCritic",
    "BernoulliActorCritic",
]
