"""双时间尺度 Cache/UAV PPO 策略及其 masked categorical 分布。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - 无 PyTorch 环境仅允许查看配置
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CategoricalSample:
    """一次合法 categorical 抽样。"""

    selected_index: int
    log_prob: Any
    entropy: Any
    probabilities: np.ndarray
    masked_logits: Any
    selected_group: str


@dataclass(frozen=True)
class MacroActionSample:
    """Cache candidate 与 duration 组成的单个 Macro 联合动作。"""

    cache_index: int
    duration_index: int
    requested_duration: int
    log_prob: Any
    entropy: Any
    value: Any
    cache_mask: np.ndarray
    duration_mask: np.ndarray
    cache_probabilities: np.ndarray
    raw_cache_probabilities: np.ndarray
    conditional_update_probabilities: np.ndarray
    cache_entropy: Any
    trigger_entropy: Any
    duration_entropy: Any
    cache_selected_group: str
    trigger_log_prob: Any
    conditional_update_log_prob: Any
    p_keep: Any
    p_update: Any
    log_p_keep: Any
    log_p_update: Any
    duration_log_prob: Any
    deterministic_cache_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class MacroComponentsEvaluation:
    """Hierarchical Cache-trigger, conditional candidate, and Duration evaluation."""

    cache_log_prob: Any
    duration_log_prob: Any
    trigger_log_prob: Any
    conditional_update_log_prob: Any
    p_keep: Any
    p_update: Any
    log_p_keep: Any
    log_p_update: Any
    joint_log_prob: Any
    cache_entropy: Any
    trigger_entropy: Any
    conditional_update_entropy: Any
    duration_entropy: Any
    value: Any
    joint_logprob_decomposition_error: Any


@dataclass(frozen=True)
class JointMicroEvaluation:
    """两个 UAV 条件式联合动作的 PPO 评估结果。"""

    log_prob: Any
    entropy: Any
    value: Any
    uav0_log_prob: Any
    uav1_log_prob: Any


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("双时间尺度 PPO 需要 PyTorch。")


def _masked_categorical(logits: Any, mask: Any) -> Any:
    """在构造分布前应用 rollout 中的合法动作 mask。"""

    valid = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if tuple(valid.shape) != tuple(logits.shape) or not bool(torch.any(valid).item()):
        raise ValueError("categorical mask 必须与 logits 对齐且至少允许一个动作。")
    if not bool(torch.all(torch.isfinite(logits[valid])).item()):
        raise FloatingPointError("categorical logits 包含 NaN/Inf。")
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~valid, -float("inf"))
    )


def _group_aware_deterministic_candidate(
    raw_scores: Any,
    probabilities: Any,
    candidate_mask: Any,
    group_ids: Any,
) -> tuple[int, int]:
    """Select a group by probability mass, then its raw-score candidate."""

    scores = torch.as_tensor(raw_scores).reshape(-1)
    probs = torch.as_tensor(probabilities, dtype=scores.dtype, device=scores.device).reshape(-1)
    mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=scores.device).reshape(-1)
    groups = torch.as_tensor(group_ids, dtype=torch.long, device=scores.device).reshape(-1)
    if not (scores.shape == probs.shape == mask.shape == groups.shape) or not len(scores):
        raise ValueError("group-aware candidate tensors must be non-empty and aligned")
    if not bool(torch.any(mask).item()) or bool(torch.any(groups[mask] > 1).item()) or bool(
        torch.any(groups[mask] < 0).item()
    ):
        raise ValueError("group-aware candidate mask/group ids are invalid")
    if not bool(torch.all(torch.isfinite(scores[mask])).item()) or not bool(
        torch.all(torch.isfinite(probs[mask])).item()
    ):
        raise FloatingPointError("group-aware scores/probabilities must be finite")

    group_mass = torch.zeros(2, dtype=probs.dtype, device=probs.device)
    group_mass.scatter_add_(0, groups[mask], probs[mask])
    selected_group = int(torch.argmax(group_mass).item())
    if selected_group == 0:
        if not bool(mask[0].item()) or int(groups[0].item()) != 0:
            raise ValueError("the KEEP/STAY group must contain legal candidate zero")
        return 0, selected_group

    selected_mask = mask & (groups == selected_group)
    if not bool(torch.any(selected_mask).item()):
        raise ValueError("selected action group has no legal candidates")
    selected_index = int(scores.masked_fill(~selected_mask, -float("inf")).argmax().item())
    return selected_index, selected_group


def _group_balance_candidate_logits(
    raw_scores: Any,
    candidate_features: Any,
    candidate_masks: Any,
    operation_slice: slice,
) -> Any:
    """复用现有 PPO 的候选组均衡规则并屏蔽非法候选。"""

    scores = torch.as_tensor(raw_scores)
    features = torch.as_tensor(candidate_features, dtype=torch.float32, device=scores.device)
    masks = torch.as_tensor(candidate_masks, dtype=torch.bool, device=scores.device)
    if features.ndim != scores.ndim + 1 or tuple(features.shape[:-1]) != tuple(scores.shape):
        raise ValueError("candidate features 与 logits 维度不匹配。")
    if tuple(masks.shape) != tuple(scores.shape) or not bool(
        torch.all(torch.any(masks, dim=-1)).item()
    ):
        raise ValueError("每个 candidate head 至少要有一个合法动作。")
    start = 0 if operation_slice.start is None else int(operation_slice.start)
    stop = int(features.shape[-1]) if operation_slice.stop is None else int(operation_slice.stop)
    groups = features[..., start:stop]
    valid_groups = groups[masks]
    if stop <= start or not bool(torch.all((valid_groups == 0.0) | (valid_groups == 1.0)).item()):
        raise ValueError("合法 candidate group 标记必须是 one-hot。")
    if not bool(torch.all(valid_groups.sum(dim=-1) == 1.0).item()):
        raise ValueError("每个合法 candidate 必须属于一个动作组。")
    group_ids = groups.argmax(dim=-1)
    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_masks = masks.reshape_as(flat_scores)
    flat_groups = group_ids.reshape_as(flat_scores)
    balanced = flat_scores.clone()
    for row in range(int(flat_scores.shape[0])):
        counts = torch.zeros(stop - start, dtype=torch.long, device=scores.device)
        counts.scatter_add_(
            0,
            flat_groups[row][flat_masks[row]],
            torch.ones_like(flat_groups[row][flat_masks[row]], dtype=torch.long),
        )
        balanced[row] -= torch.log(counts[flat_groups[row]].clamp_min(1).to(scores.dtype))
    return balanced.reshape(scores.shape).masked_fill(~masks, -float("inf"))


def _mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
    *,
    activation: Any = None,
) -> Any:
    layers: list[Any] = []
    width = int(input_dim)
    activation_factory = nn.Tanh if activation is None else activation
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, int(hidden)), activation_factory()))
        width = int(hidden)
    layers.append(nn.Linear(width, int(output_dim)))
    return nn.Sequential(*layers)


class DualTimescaleActorCritic(nn.Module if nn is not None else object):
    """Macro Cache SMDP、独立 Long-Value Critic 与 Micro UAV PPO。"""

    architecture_id = "adaptive_dual_timescale_macro_cache_smdp_micro_uav_ppo"
    architecture_version = 23
    cardinality_corrected_unified_architecture_id = (
        "adaptive_dual_timescale_proposed_cache_action_cardinality_corrected_ppo"
    )
    cardinality_corrected_unified_architecture_version = 25

    def __init__(
        self,
        observation_dim: int,
        macro_cache_feature_dim: int,
        micro_uav_feature_dim: int,
        duration_candidates: tuple[int, ...],
        prediction_horizon: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        *,
        unified_cache_action: bool = True,
        cardinality_corrected_cache_action: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.macro_cache_feature_dim = int(macro_cache_feature_dim)
        self.micro_uav_feature_dim = int(micro_uav_feature_dim)
        self.duration_candidates = tuple(int(value) for value in duration_candidates)
        self.prediction_horizon = int(prediction_horizon)
        self.unified_cache_action = bool(unified_cache_action)
        self.cardinality_corrected_cache_action = bool(
            cardinality_corrected_cache_action
        )
        if not self.unified_cache_action or not self.cardinality_corrected_cache_action:
            raise ValueError(
                "The PROPOSED policy supports only the unified Cache candidate action with cardinality correction."
            )
        self.architecture_id = self.cardinality_corrected_unified_architecture_id
        self.architecture_version = self.cardinality_corrected_unified_architecture_version
        self.trigger_observation_mean = np.zeros(self.observation_dim, dtype=np.float32)
        self.trigger_observation_std = np.ones(self.observation_dim, dtype=np.float32)
        self.trigger_observation_raw_std = np.ones(self.observation_dim, dtype=np.float32)
        self.trigger_observation_normalization_active = False
        self.trigger_observation_normalization_metadata: dict[str, Any] = {}
        self.trigger_observation_input_clip: tuple[float, float] | None = None
        self.trigger_observation_low_variance_feature_count = 0
        sizes = tuple(int(value) for value in hidden_sizes)
        if min(self.observation_dim, self.macro_cache_feature_dim, self.micro_uav_feature_dim) <= 0:
            raise ValueError("策略 observation/candidate feature dimensions 必须为正数。")
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("hidden_sizes 必须是正整数序列。")
        if (
            self.prediction_horizon <= 0
            or not self.duration_candidates
            or any(value <= 0 for value in self.duration_candidates)
            or tuple(sorted(set(self.duration_candidates))) != self.duration_candidates
            or self.duration_candidates[-1] > self.prediction_horizon
        ):
            raise ValueError("Macro duration candidates 与 prediction horizon 不匹配。")
        hidden = sizes[-1]

        self.macro_actor_encoder = _mlp(self.observation_dim, sizes[:-1], hidden)
        if len(sizes) == 1:
            self.macro_actor_encoder = nn.Sequential(
                nn.Linear(self.observation_dim, hidden), nn.Tanh()
            )
        self.macro_cache_candidate_encoder = nn.Sequential(
            nn.Linear(self.macro_cache_feature_dim, hidden), nn.Tanh()
        )
        self.macro_cache_scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.macro_duration_scorer = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.macro_critic = _mlp(
            self.observation_dim, sizes, 1, activation=nn.SiLU
        )

        self.micro_actor_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, hidden), nn.Tanh()
        )
        self.micro_uav_candidate_scorer = nn.Sequential(
            nn.Linear(hidden + self.micro_uav_feature_dim + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.micro_critic = _mlp(
            self.observation_dim, sizes, 1, activation=nn.SiLU
        )

        self.long_value_critic = _mlp(
            self.observation_dim, (128, 128), 1, activation=nn.SiLU
        )

        # 该模块暂时仅为当前 checkpoint 状态结构保留，正式策略不会调用。
        self.macro_cache_trigger_actor = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.macro_cache_trigger_actor[-1].weight)
        nn.init.zeros_(self.macro_cache_trigger_actor[-1].bias)

    def _macro_critic_inputs(self, observations: Any) -> Any:
        return 2.0 * observations - 1.0

    def _micro_critic_inputs(self, observations: Any) -> Any:
        return 2.0 * observations - 1.0

    @staticmethod
    def _observation_batch(observations: Any, device: Any) -> tuple[Any, bool]:
        values = torch.as_tensor(observations, dtype=torch.float32, device=device)
        single = values.ndim == 1
        if single:
            values = values.unsqueeze(0)
        if values.ndim != 2 or not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError("observations 必须是有限的 [B,D] 张量。")
        return values, single




    def trigger_observation_normalization_payload(self) -> dict[str, Any]:
        return {
            "active": bool(self.trigger_observation_normalization_active),
            "mean": self.trigger_observation_mean.astype(np.float32, copy=True),
            "std": self.trigger_observation_std.astype(np.float32, copy=True),
            "raw_mean": self.trigger_observation_mean.astype(np.float32, copy=True),
            "raw_std": self.trigger_observation_raw_std.astype(np.float32, copy=True),
            "scale": self.trigger_observation_std.astype(np.float32, copy=True),
            "metadata": dict(self.trigger_observation_normalization_metadata),
        }

    @staticmethod
    def _candidate_batch(
        features: Any,
        masks: Any,
        batch_size: int,
        feature_dim: int,
        device: Any,
    ) -> tuple[Any, Any]:
        feature_values = torch.as_tensor(features, dtype=torch.float32, device=device)
        mask_values = torch.as_tensor(masks, dtype=torch.bool, device=device)
        if feature_values.ndim == 2:
            feature_values = feature_values.unsqueeze(0)
        if mask_values.ndim == 1:
            mask_values = mask_values.unsqueeze(0)
        if (
            feature_values.ndim != 3
            or int(feature_values.shape[0]) != int(batch_size)
            or int(feature_values.shape[-1]) != int(feature_dim)
            or tuple(mask_values.shape) != tuple(feature_values.shape[:2])
        ):
            raise ValueError("candidate features/masks shape 不匹配。")
        if not bool(torch.all(torch.isfinite(feature_values)).item()):
            raise ValueError("candidate features 包含 NaN/Inf。")
        if not bool(torch.all(torch.any(mask_values, dim=-1)).item()):
            raise ValueError("每个候选集合必须保留至少一个合法动作。")
        return feature_values, mask_values

    def macro_value(self, observations: Any) -> Any:
        obs, single = self._observation_batch(
            observations, next(self.parameters()).device
        )
        values = self.macro_critic(self._macro_critic_inputs(obs)).squeeze(-1)
        return values.squeeze(0) if single else values

    def micro_value(self, observations: Any) -> Any:
        obs, single = self._observation_batch(observations, next(self.parameters()).device)
        values = self.micro_critic(self._micro_critic_inputs(obs)).squeeze(-1)
        return values.squeeze(0) if single else values

    def long_value(self, observations: Any) -> Any:
        """Evaluate the independent Long Critic on the raw Macro observation."""

        obs, single = self._observation_batch(
            observations, next(self.parameters()).device
        )
        if int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Long-Value Critic observation dimension mismatch.")
        values = self.long_value_critic(obs).squeeze(-1)
        return values.squeeze(0) if single else values

    @staticmethod
    def _balance_macro_cache_group_mass(raw_scores: Any, candidate_masks: Any) -> Any:
        scores = torch.as_tensor(raw_scores)
        candidate_indices = torch.arange(int(scores.shape[-1]), device=scores.device)
        group_features = torch.stack(
            (candidate_indices == 0, candidate_indices > 0), dim=-1
        ).to(dtype=torch.float32)
        group_features = group_features.reshape(
            *([1] * (scores.ndim - 1)), int(scores.shape[-1]), 2
        ).expand(*scores.shape, 2)
        return _group_balance_candidate_logits(
            scores, group_features, candidate_masks, slice(0, 2)
        )

    @staticmethod
    def _cardinality_corrected_cache_logits(raw_logits: Any, candidate_masks: Any) -> Any:
        """Subtract log of the legal UPDATE cardinality from each legal UPDATE logit."""

        logits = torch.as_tensor(raw_logits)
        masks = torch.as_tensor(candidate_masks, dtype=torch.bool, device=logits.device)
        if logits.ndim < 1 or tuple(masks.shape) != tuple(logits.shape) or logits.shape[-1] < 1:
            raise ValueError("Unified Cache logits and legal masks must be non-empty and aligned.")
        if not bool(torch.all(masks[..., 0]).item()):
            raise ValueError("Unified Cache candidate 0 must remain legal KEEP.")
        if not bool(torch.all(torch.isfinite(logits[masks])).item()):
            raise FloatingPointError("Unified Cache candidate logits contain NaN/Inf.")
        legal_update_count = masks[..., 1:].sum(dim=-1)
        log_cardinality = torch.log(
            legal_update_count.clamp_min(1).to(dtype=logits.dtype)
        ).unsqueeze(-1)
        corrected = logits.clone()
        corrected[..., 1:] = torch.where(
            masks[..., 1:],
            logits[..., 1:] - log_cardinality,
            logits[..., 1:],
        )
        return corrected

    def _macro_logits(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        duration_masks: Any,
        cache_actions: Any | None = None,
        duration_actions: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        device = next(self.parameters()).device
        obs, single = self._observation_batch(observations, device)
        if int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Macro observation dimension mismatch。")
        features, masks = self._candidate_batch(
            cache_candidate_features,
            cache_candidate_masks,
            int(obs.shape[0]),
            self.macro_cache_feature_dim,
            device,
        )
        duration_mask = torch.as_tensor(duration_masks, dtype=torch.bool, device=device)
        if duration_mask.ndim == 1:
            duration_mask = duration_mask.unsqueeze(0)
        expected_duration_shape = (int(obs.shape[0]), len(self.duration_candidates))
        if tuple(duration_mask.shape) != expected_duration_shape:
            raise ValueError("duration_masks shape mismatch。")
        if not bool(torch.all(torch.any(duration_mask, dim=-1)).item()):
            raise ValueError("duration mask 至少要允许一个时长。")

        hidden = self.macro_actor_encoder(obs)
        candidate_hidden = self.macro_cache_candidate_encoder(features)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, int(features.shape[1]), -1)
        cache_logits = self.macro_cache_scorer(
            torch.cat((expanded_hidden, candidate_hidden), dim=-1)
        ).squeeze(-1)
        cache_logits = self._cardinality_corrected_cache_logits(cache_logits, masks)

        if cache_actions is None:
            raise ValueError("Macro duration probability requires selected Cache candidates。")
        cache_action = torch.as_tensor(cache_actions, dtype=torch.long, device=device).reshape(-1)
        if tuple(cache_action.shape) != (int(obs.shape[0]),):
            raise ValueError("cache_actions shape mismatch。")
        if bool(torch.any(cache_action < 0).item()) or bool(
            torch.any(cache_action >= int(features.shape[1])).item()
        ):
            raise ValueError("Macro cache action index 越界。")
        if not bool(torch.all(masks.gather(1, cache_action[:, None]).squeeze(1)).item()):
            raise ValueError("Macro cache action 被 rollout mask 屏蔽。")
        if bool(torch.any((cache_action > 0) & ~torch.any(masks[:, 1:], dim=-1)).item()):
            raise ValueError("Macro UPDATE action requires at least one legal UPDATE candidate。")
        selected_candidate = candidate_hidden[
            torch.arange(int(obs.shape[0]), device=device), cache_action
        ]
        duration_values = torch.as_tensor(
            self.duration_candidates, dtype=torch.float32, device=device
        ) / float(self.prediction_horizon)
        duration_values = duration_values.reshape(1, -1, 1).expand(
            int(obs.shape[0]), -1, -1
        )
        duration_hidden = torch.cat(
            (
                hidden.unsqueeze(1).expand(-1, len(self.duration_candidates), -1),
                selected_candidate.unsqueeze(1).expand(-1, len(self.duration_candidates), -1),
                duration_values,
            ),
            dim=-1,
        )
        duration_logits = self.macro_duration_scorer(duration_hidden).squeeze(-1)
        duration_dist = _masked_categorical(duration_logits, duration_mask)
        if duration_actions is not None:
            duration_action = torch.as_tensor(
                duration_actions, dtype=torch.long, device=device
            ).reshape(-1)
            if tuple(duration_action.shape) != (int(obs.shape[0]),):
                raise ValueError("duration_actions shape mismatch。")
            if bool(torch.any(duration_action < 0).item()) or bool(
                torch.any(duration_action >= len(self.duration_candidates)).item()
            ):
                raise ValueError("Macro duration action index 越界。")
            if not bool(
                torch.all(
                    duration_mask.gather(1, duration_action[:, None]).squeeze(1)
                ).item()
            ):
                raise ValueError("Macro duration action 被 rollout mask 屏蔽。")
        values = self.macro_value(obs)
        return cache_logits, duration_dist.logits, values, masks, duration_mask, cache_action

    def sample_macro(
        self,
        observation: Any,
        cache_candidate_features: Any,
        cache_candidate_mask: Any,
        *,
        deterministic: bool = False,
    ) -> MacroActionSample:
        """Sample stochastic rollout actions or deterministic evaluation actions.

        Stochastic mode samples both Cache and Duration categoricals. Deterministic
        mode uses group-aware Cache selection and the highest-probability Duration.
        """
        features = np.asarray(cache_candidate_features, dtype=np.float32)
        cache_mask = np.asarray(cache_candidate_mask, dtype=bool).reshape(-1)
        if features.ndim != 2 or len(features) != len(cache_mask) or not len(cache_mask):
            raise ValueError("Macro cache candidates/mask must be non-empty and aligned。")
        if not bool(cache_mask[0]):
            raise ValueError("Macro candidate 0 must remain the legal KEEP action.")
        device = next(self.parameters()).device
        durations_mask = np.ones(len(self.duration_candidates), dtype=bool)
        with torch.no_grad():
            obs, _ = self._observation_batch(observation, device)
            padded_features = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            cache_mask_t = torch.as_tensor(cache_mask, dtype=torch.bool, device=device).unsqueeze(0)
            hidden = self.macro_actor_encoder(obs)
            embedded = self.macro_cache_candidate_encoder(padded_features)
            candidate_logits = self.macro_cache_scorer(
                torch.cat((hidden.unsqueeze(1).expand(-1, len(features), -1), embedded), dim=-1)
            ).squeeze(-1).squeeze(0)

            raw_candidate_logits = candidate_logits
            raw_cache_dist = _masked_categorical(candidate_logits, cache_mask_t[0])
            candidate_logits = self._cardinality_corrected_cache_logits(
                candidate_logits, cache_mask_t[0]
            )
            cache_dist = _masked_categorical(candidate_logits, cache_mask_t[0])
            deterministic_cache_diagnostics = None
            group_ids = torch.ones_like(cache_mask_t[0], dtype=torch.long)
            group_ids[0] = 0
            if deterministic:
                cache_index, selected_cache_group = _group_aware_deterministic_candidate(
                    raw_candidate_logits,
                    cache_dist.probs,
                    cache_mask_t[0],
                    group_ids,
                )
                legacy_flat_index = int(torch.argmax(cache_dist.logits).item())
                legacy_reference_logits = self._cardinality_corrected_cache_logits(
                    raw_candidate_logits, cache_mask_t[0]
                )
                legacy_reference_dist = _masked_categorical(
                    legacy_reference_logits, cache_mask_t[0]
                )
                probability_max_difference = float(
                    torch.max(torch.abs(cache_dist.probs - legacy_reference_dist.probs))
                    .detach()
                    .cpu()
                )
                probability_sum_error = float(
                    torch.abs(cache_dist.probs.sum() - 1.0).detach().cpu()
                )
                update_mask = cache_mask_t[0] & (group_ids == 1)
                legal_update_count = int(update_mask.sum().item())
                legal_update_best_score = (
                    float(raw_candidate_logits.masked_fill(~update_mask, -float("inf")).max().cpu())
                    if legal_update_count
                    else None
                )
                selected_raw_score = float(raw_candidate_logits[cache_index].detach().cpu())
                selected_candidate_is_legal = bool(cache_mask_t[0, cache_index].item())
                update_inner_argmax_violation = bool(
                    selected_cache_group == 1
                    and (
                        cache_index <= 0
                        or not selected_candidate_is_legal
                        or selected_raw_score != legal_update_best_score
                    )
                )
                p_keep_value = float(cache_dist.probs[0].detach().cpu())
                p_update_group_value = float(cache_dist.probs[1:].sum().detach().cpu())
                deterministic_cache_diagnostics = {
                    "p_keep": p_keep_value,
                    "p_update_group": p_update_group_value,
                    "new_selected_group": "KEEP" if selected_cache_group == 0 else "UPDATE",
                    "new_selected_candidate_index": cache_index,
                    "legacy_flat_selected_candidate_index": legacy_flat_index,
                    "legacy_flat_selected_group": "KEEP" if legacy_flat_index == 0 else "UPDATE",
                    "old_new_disagreement": cache_index != legacy_flat_index,
                    "group_tie_break": "KEEP",
                    "legal_update_candidate_count": legal_update_count,
                    "selected_candidate_is_legal": selected_candidate_is_legal,
                    "selected_raw_score": selected_raw_score,
                    "legal_update_best_raw_score": legal_update_best_score,
                    "update_inner_argmax_violation": update_inner_argmax_violation,
                    "probability_sum_error": probability_sum_error,
                    "probability_max_difference_vs_legacy_formula": probability_max_difference,
                }
            else:
                cache_index = int(cache_dist.sample().item())
                selected_cache_group = 0 if cache_index == 0 else 1
            selected_embed = embedded[0, cache_index]
            duration_values = torch.as_tensor(
                self.duration_candidates, dtype=torch.float32, device=device
            ) / float(self.prediction_horizon)
            duration_input = torch.cat(
                (
                    hidden.expand(len(self.duration_candidates), -1),
                    selected_embed.expand(len(self.duration_candidates), -1),
                    duration_values.reshape(-1, 1),
                ),
                dim=-1,
            )
            duration_logits = self.macro_duration_scorer(duration_input).squeeze(-1)
            duration_dist = _masked_categorical(
                duration_logits,
                torch.as_tensor(durations_mask, dtype=torch.bool, device=device),
            )
            duration_index = (
                int(torch.argmax(duration_dist.logits).item())
                if deterministic
                else int(duration_dist.sample().item())
            )
            cache_tensor = torch.as_tensor(cache_index, dtype=torch.long, device=device)
            duration_tensor = torch.as_tensor(duration_index, dtype=torch.long, device=device)
            cache_probabilities = cache_dist.probs
            probability_sum = cache_probabilities.sum()
            if (
                not bool(torch.isfinite(cache_probabilities).all().item())
                or not torch.allclose(
                    probability_sum,
                    probability_sum.new_ones(()),
                    atol=1.0e-6,
                    rtol=0.0,
                )
            ):
                raise FloatingPointError("Unified Cache Candidate probabilities are invalid.")
            cache_log_prob = cache_dist.log_prob(cache_tensor)
            duration_log_prob = duration_dist.log_prob(duration_tensor)
            cache_entropy = cache_dist.entropy()
            duration_entropy = duration_dist.entropy()
            zero = cache_log_prob.new_zeros(())
            p_keep = cache_probabilities[0]
            p_update = cache_probabilities[1:].sum()
            log_p_keep = torch.log(p_keep.clamp_min(1.0e-30))
            log_p_update = torch.log(p_update.clamp_min(1.0e-30))
            return MacroActionSample(
                cache_index=cache_index,
                duration_index=duration_index,
                requested_duration=self.duration_candidates[duration_index],
                log_prob=cache_log_prob + duration_log_prob,
                entropy=cache_entropy + duration_entropy,
                value=self.macro_value(obs).squeeze(),
                cache_mask=cache_mask.copy(),
                duration_mask=durations_mask,
                cache_probabilities=cache_probabilities.detach().cpu().numpy().copy(),
                raw_cache_probabilities=(
                    cache_probabilities.detach().cpu().numpy().copy()
                    if raw_cache_dist is None
                    else raw_cache_dist.probs.detach().cpu().numpy().copy()
                ),
                conditional_update_probabilities=cache_probabilities.detach().cpu().numpy().copy(),
                cache_entropy=cache_entropy,
                trigger_entropy=zero,
                duration_entropy=duration_entropy,
                cache_selected_group="KEEP" if selected_cache_group == 0 else "UPDATE",
                trigger_log_prob=zero,
                conditional_update_log_prob=cache_log_prob,
                p_keep=p_keep,
                p_update=p_update,
                log_p_keep=log_p_keep,
                log_p_update=log_p_update,
                duration_log_prob=duration_log_prob,
                deterministic_cache_diagnostics=deterministic_cache_diagnostics,
            )



    def evaluate_macro(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_actions: Any,
        duration_masks: Any,
        duration_actions: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        components = self.evaluate_macro_components(
            observations,
            cache_candidate_features,
            cache_candidate_masks,
            cache_actions,
            duration_masks,
            duration_actions,
        )
        return (
            components.joint_log_prob,
            components.cache_entropy + components.duration_entropy,
            components.value,
            components.cache_log_prob,
            components.duration_log_prob,
        )

    def evaluate_macro_components(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_actions: Any,
        duration_masks: Any,
        duration_actions: Any,
    ) -> MacroComponentsEvaluation:
        """Evaluate the formal joint Cache/Duration action and its factors."""

        cache_logits, duration_logits, values, masks, d_masks, cache_actions_t = self._macro_logits(
            observations,
            cache_candidate_features,
            cache_candidate_masks,
            duration_masks,
            cache_actions,
            duration_actions,
        )
        device = cache_logits.device
        observations_t, _ = self._observation_batch(observations, device)
        duration_actions_t = torch.as_tensor(duration_actions, dtype=torch.long, device=device).reshape(-1)
        duration_dist = _masked_categorical(duration_logits, d_masks)
        duration_log_prob = duration_dist.log_prob(duration_actions_t)

        cache_dist = _masked_categorical(cache_logits, masks)
        cache_log_prob = cache_dist.log_prob(cache_actions_t)
        cache_probabilities = cache_dist.probs
        cache_entropy = cache_dist.entropy()
        p_keep = cache_probabilities[:, 0]
        p_update = cache_probabilities[:, 1:].sum(dim=-1)
        log_p_keep = torch.log(p_keep.clamp_min(1.0e-30))
        log_p_update = torch.log(p_update.clamp_min(1.0e-30))
        zero = torch.zeros_like(cache_log_prob)
        joint_log_prob = cache_log_prob + duration_log_prob
        decomposition_error = torch.abs(
            joint_log_prob.to(torch.float64)
            - cache_log_prob.to(torch.float64)
            - duration_log_prob.to(torch.float64)
        )
        if not bool(torch.isfinite(cache_probabilities).all().item()) or not torch.allclose(
            cache_probabilities.sum(dim=-1),
            torch.ones_like(cache_probabilities.sum(dim=-1)),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise FloatingPointError("Unified Cache Candidate probabilities are invalid.")
        return MacroComponentsEvaluation(
            cache_log_prob=cache_log_prob,
            duration_log_prob=duration_log_prob,
            trigger_log_prob=zero,
            conditional_update_log_prob=cache_log_prob,
            p_keep=p_keep,
            p_update=p_update,
            log_p_keep=log_p_keep,
            log_p_update=log_p_update,
            joint_log_prob=joint_log_prob,
            cache_entropy=cache_entropy,
            trigger_entropy=zero,
            conditional_update_entropy=cache_entropy,
            duration_entropy=duration_dist.entropy(),
            value=values,
            joint_logprob_decomposition_error=decomposition_error,
        )

    def _micro_candidate_raw_logits(
        self,
        observation: Any,
        candidate_features: Any,
        candidate_mask: Any,
        uav0_position_condition: Any,
    ) -> Any:
        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observation, device)
        if int(obs.shape[0]) != 1 or int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Micro candidate sampling requires one state。")
        features = torch.as_tensor(candidate_features, dtype=torch.float32, device=device)
        mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=device).reshape(-1)
        condition = torch.as_tensor(
            uav0_position_condition, dtype=torch.float32, device=device
        ).reshape(2)
        if (
            features.ndim != 2
            or int(features.shape[-1]) != self.micro_uav_feature_dim
            or len(features) != len(mask)
            or not len(mask)
        ):
            raise ValueError("Micro candidate features/mask shape mismatch。")
        if not bool(torch.all(torch.isfinite(features)).item()) or not bool(
            torch.all(torch.isfinite(condition)).item()
        ):
            raise ValueError("Micro candidate inputs must be finite。")
        hidden = self.micro_actor_encoder(obs).expand(len(features), -1)
        inputs = torch.cat(
            (hidden, features, condition.expand(len(features), -1)), dim=-1
        )
        return self.micro_uav_candidate_scorer(inputs).squeeze(-1)

    def _micro_candidate_logits(
        self,
        observation: Any,
        candidate_features: Any,
        candidate_mask: Any,
        uav0_position_condition: Any,
    ) -> Any:
        scores = self._micro_candidate_raw_logits(
            observation,
            candidate_features,
            candidate_mask,
            uav0_position_condition,
        )
        features = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=scores.device
        )
        mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=scores.device).reshape(-1)
        return _group_balance_candidate_logits(
            scores,
            features,
            mask,
            slice(self.micro_uav_feature_dim - 2, self.micro_uav_feature_dim),
        )

    def sample_micro_head(
        self,
        observation: Any,
        candidate_features: Any,
        candidate_mask: Any,
        *,
        uav_index: int,
        uav0_position_condition: Any,
        deterministic: bool = False,
    ) -> CategoricalSample:
        if int(uav_index) not in (0, 1):
            raise ValueError("当前 Micro 策略要求固定 UAV0、UAV1 顺序。")
        with torch.no_grad():
            raw_logits = self._micro_candidate_raw_logits(
                observation,
                candidate_features,
                candidate_mask,
                uav0_position_condition,
            )
            features = torch.as_tensor(
                candidate_features, dtype=torch.float32, device=raw_logits.device
            )
            mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=raw_logits.device)
            logits = _group_balance_candidate_logits(
                raw_logits,
                features,
                mask,
                slice(self.micro_uav_feature_dim - 2, self.micro_uav_feature_dim),
            )
            distribution = _masked_categorical(logits, mask)
            group_ids = features[:, -2:].argmax(dim=-1)
            if deterministic:
                selected, selected_group = _group_aware_deterministic_candidate(
                    raw_logits, distribution.probs, mask, group_ids
                )
            else:
                selected = int(distribution.sample().item())
                selected_group = int(group_ids[selected].item())
            selected_t = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
            return CategoricalSample(
                selected_index=selected,
                log_prob=distribution.log_prob(selected_t),
                entropy=distribution.entropy(),
                probabilities=distribution.probs.detach().cpu().numpy().copy(),
                masked_logits=distribution.logits,
                selected_group="STAY" if selected_group == 0 else "MOVE",
            )

    def evaluate_micro_joint(
        self,
        observations: Any,
        candidate_features: Any,
        candidate_masks: Any,
        actions: Any,
        uav0_position_conditions: Any,
    ) -> JointMicroEvaluation:
        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observations, device)
        features = torch.as_tensor(candidate_features, dtype=torch.float32, device=device)
        masks = torch.as_tensor(candidate_masks, dtype=torch.bool, device=device)
        action_values = torch.as_tensor(actions, dtype=torch.long, device=device)
        conditions = torch.as_tensor(
            uav0_position_conditions, dtype=torch.float32, device=device
        )
        if features.ndim != 4 or tuple(features.shape[:2]) != (int(obs.shape[0]), 2):
            raise ValueError("Micro rollout features must be [B,2,C,F]。")
        if int(features.shape[-1]) != self.micro_uav_feature_dim:
            raise ValueError("Micro candidate feature dimension mismatch。")
        if tuple(masks.shape) != tuple(features.shape[:3]):
            raise ValueError("Micro rollout masks must match candidate features。")
        if tuple(action_values.shape) != (int(obs.shape[0]), 2):
            raise ValueError("Micro rollout actions must be [B,2]。")
        if tuple(conditions.shape) != (int(obs.shape[0]), 2):
            raise ValueError("UAV0 conditional positions must be [B,2]。")
        if not bool(torch.all(torch.isfinite(features)).item()) or not bool(
            torch.all(torch.isfinite(conditions)).item()
        ):
            raise ValueError("Micro rollout candidate data contains NaN/Inf。")
        hidden = self.micro_actor_encoder(obs)
        head_log_probs: list[Any] = []
        head_entropies: list[Any] = []
        for head in range(2):
            head_features = features[:, head]
            head_masks = masks[:, head]
            condition = (
                torch.zeros_like(conditions)
                if head == 0
                else conditions
            )
            expanded_hidden = hidden.unsqueeze(1).expand(-1, int(features.shape[2]), -1)
            logits = self.micro_uav_candidate_scorer(
                torch.cat(
                    (
                        expanded_hidden,
                        head_features,
                        condition.unsqueeze(1).expand(-1, int(features.shape[2]), -1),
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            balanced = _group_balance_candidate_logits(
                logits,
                head_features,
                head_masks,
                slice(self.micro_uav_feature_dim - 2, self.micro_uav_feature_dim),
            )
            dist = _masked_categorical(balanced, head_masks)
            action = action_values[:, head]
            valid = head_masks.gather(1, action[:, None]).squeeze(1)
            if not bool(torch.all(valid).item()):
                raise ValueError("PPO action 被 rollout 当时保存的 mask 屏蔽。")
            head_log_probs.append(dist.log_prob(action))
            head_entropies.append(dist.entropy())
        values = self.micro_critic(self._micro_critic_inputs(obs)).squeeze(-1)
        log0, log1 = head_log_probs
        entropies = head_entropies[0] + head_entropies[1]
        return JointMicroEvaluation(
            log_prob=log0 + log1,
            entropy=entropies,
            value=values,
            uav0_log_prob=log0,
            uav1_log_prob=log1,
        )

    def macro_actor_parameters(self) -> tuple[Any, ...]:
        modules = (
            self.macro_actor_encoder,
            self.macro_cache_candidate_encoder,
            self.macro_cache_scorer,
            self.macro_duration_scorer,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

    def macro_cache_trigger_actor_parameters(self) -> tuple[Any, ...]:
        return tuple(self.macro_cache_trigger_actor.parameters())

    def macro_critic_parameters(self) -> tuple[Any, ...]:
        return tuple(self.macro_critic.parameters())

    def long_value_critic_parameters(self) -> tuple[Any, ...]:
        return tuple(self.long_value_critic.parameters())

    def micro_actor_parameters(self) -> tuple[Any, ...]:
        modules = (self.micro_actor_encoder, self.micro_uav_candidate_scorer)
        return tuple(parameter for module in modules for parameter in module.parameters())

    def micro_critic_parameters(self) -> tuple[Any, ...]:
        return tuple(self.micro_critic.parameters())

    def macro_parameters(self) -> tuple[Any, ...]:
        return (*self.macro_actor_parameters(), *self.macro_critic_parameters())

    def micro_parameters(self) -> tuple[Any, ...]:
        return (*self.micro_actor_parameters(), *self.micro_critic_parameters())

    def macro_state_dict(self) -> dict[str, Any]:
        prefixes = (
            "macro_actor_encoder.",
            "macro_cache_candidate_encoder.",
            "macro_cache_scorer.",
            "macro_duration_scorer.",
            "macro_critic.",
        )
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def micro_state_dict(self) -> dict[str, Any]:
        prefixes = (
            "micro_actor_encoder.",
            "micro_uav_candidate_scorer.",
            "micro_critic.",
        )
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def long_value_critic_state_dict(self) -> dict[str, Any]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.long_value_critic.state_dict().items()
        }

    def n2_critic_state_dict(self) -> dict[str, Any]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.macro_critic.state_dict().items()
        }

    def load_separated_state_dicts(
        self,
        macro_state: dict[str, Any],
        micro_state: dict[str, Any],
        long_value_critic_state: dict[str, Any],
        macro_cache_trigger_actor_state: dict[str, Any],
    ) -> None:
        long_state = {
            f"long_value_critic.{key}": value
            for key, value in long_value_critic_state.items()
        }
        trigger_state = {
            f"macro_cache_trigger_actor.{key}": value
            for key, value in macro_cache_trigger_actor_state.items()
        }
        combined = {**macro_state, **micro_state, **long_state, **trigger_state}
        self.load_state_dict(combined, strict=True)
