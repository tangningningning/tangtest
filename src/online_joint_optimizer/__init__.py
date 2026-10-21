"""Prediction-driven, bounded cache and UAV joint optimization core."""

from .cache_optimizer import (
    CacheCandidateScore,
    CacheOptimizationResult,
    CacheOptimizer,
    ForecastTailResult,
)
from .candidate_evaluator import (
    CandidateEvaluation,
    CandidateEvaluator,
    EvaluationBudget,
    PredictedSlotInput,
    SearchBudget,
    SlotInput,
    SystemModelContext,
)
from .joint_optimizer import JointOptimizationResult, JointOptimizer
from .prediction_adapter import (
    PredictionAdaptationResult,
    adapt_aligned_rollout_step,
    adapt_prediction_origin,
)
from .uav_scheduler import UavCandidateEvaluation, UavScheduleResult, UavScheduler

__all__ = [
    "CacheCandidateScore",
    "CacheOptimizationResult",
    "CacheOptimizer",
    "CandidateEvaluation",
    "CandidateEvaluator",
    "EvaluationBudget",
    "ForecastTailResult",
    "JointOptimizationResult",
    "JointOptimizer",
    "PredictedSlotInput",
    "PredictionAdaptationResult",
    "SearchBudget",
    "SlotInput",
    "SystemModelContext",
    "UavCandidateEvaluation",
    "UavScheduleResult",
    "UavScheduler",
    "adapt_aligned_rollout_step",
    "adapt_prediction_origin",
]
