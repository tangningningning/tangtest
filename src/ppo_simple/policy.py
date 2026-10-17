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
    """单个 Macro Cache 动作的采样结果。"""

    cache_index: int
    log_prob: Any
    entropy: Any
    value: Any
    cache_mask: np.ndarray
    cache_probabilities: np.ndarray
    raw_cache_probabilities: np.ndarray
    cache_entropy: Any
    cache_selected_group: str
    p_keep: Any
    p_update: Any
    deterministic_cache_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class MacroComponentsEvaluation:
    """Cache 动作与一帧中所有采样 termination 决策的概率分解。"""

    cache_log_prob: Any
    termination_log_probs: Any
    termination_entropies: Any
    termination_probabilities: Any
    termination_log_prob_sum: Any
    termination_entropy_sum: Any
    p_keep: Any
    p_update: Any
    joint_log_prob: Any
    cache_entropy: Any
    value: Any
    joint_logprob_decomposition_error: Any
    node_cache_log_probs: Any = None
    node_cache_entropies: Any = None
    node_cache_probabilities: Any = None


@dataclass(frozen=True)
class MultiNodeMacroActionSample:
    """Nine independently sampled Cache heads sharing one state encoder."""

    cache_actions: np.ndarray
    node_log_probs: Any
    node_entropies: Any
    value: Any
    cache_masks: np.ndarray
    node_probabilities: np.ndarray
    raw_node_probabilities: np.ndarray
    node_p_keep: Any
    node_p_update: Any
    deterministic_diagnostics: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class SparseCoordinatedCacheActionSample:
    """One KEEP-or-node action and its conditional target distribution."""

    cache_actions: np.ndarray
    group_action: int
    group_log_prob: Any
    conditional_log_prob: Any
    cache_log_prob: Any
    cache_entropy: Any
    value: Any
    group_probabilities: np.ndarray
    conditional_probabilities: np.ndarray
    node_probabilities: np.ndarray
    node_p_keep: Any
    node_p_update: Any
    node_marginal_entropies: Any
    deterministic_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class TerminationActionSample:
    """单个可学习 CONTINUE/TERMINATE 决策的采样结果。"""

    selected_index: int
    log_prob: Any
    entropy: Any
    probabilities: np.ndarray


@dataclass(frozen=True)
class TerminationEvaluation:
    """批量 termination 决策的概率评估结果。"""

    log_prob: Any
    entropy: Any
    probabilities: Any


@dataclass(frozen=True)
class JointMicroEvaluation:
    """两个 UAV 条件式联合动作的 PPO 评估结果。"""

    log_prob: Any
    entropy: Any
    value: Any
    uav0_log_prob: Any
    uav1_log_prob: Any
    uav0_stay_probability: Any
    uav0_move_probability: Any
    uav0_group_entropy: Any
    uav1_stay_probability: Any
    uav1_move_probability: Any
    uav1_group_entropy: Any
    joint_group_entropy: Any
    mean_group_entropy: Any


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


def _sum_log_probabilities(log_probabilities: Any, *, dim: int) -> Any:
    """Accumulate float32 action log-probabilities in float64, then restore dtype."""

    if not log_probabilities.is_floating_point():
        raise TypeError("action log-probabilities must use a floating-point dtype")
    return log_probabilities.to(dtype=torch.float64).sum(dim=dim).to(
        dtype=log_probabilities.dtype
    )


def _macro_log_prob_from_components(
    cache_log_prob: Any, termination_log_probabilities: Any, *, dim: int
) -> tuple[Any, Any]:
    """Use the same rounded component sum and joint addition in rollout and PPO."""

    termination_log_prob_sum = _sum_log_probabilities(
        termination_log_probabilities, dim=dim
    )
    return termination_log_prob_sum, cache_log_prob + termination_log_prob_sum


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

    architecture_id = (
        "adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_v3_no_tv_local"
    )
    architecture_version = 32

    def __init__(
        self,
        observation_dim: int,
        macro_cache_feature_dim: int,
        micro_uav_feature_dim: int,
        adaptive_holding_min_slots: int,
        adaptive_holding_max_slots: int,
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
        self.adaptive_holding_min_slots = int(adaptive_holding_min_slots)
        self.adaptive_holding_max_slots = int(adaptive_holding_max_slots)
        self.unified_cache_action = bool(unified_cache_action)
        self.cardinality_corrected_cache_action = bool(
            cardinality_corrected_cache_action
        )
        if not self.cardinality_corrected_cache_action:
            raise ValueError(
                "The PROPOSED policy requires Cache candidate cardinality correction."
            )
        sizes = tuple(int(value) for value in hidden_sizes)
        if min(self.observation_dim, self.macro_cache_feature_dim, self.micro_uav_feature_dim) <= 0:
            raise ValueError("策略 observation/candidate feature dimensions 必须为正数。")
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("hidden_sizes 必须是正整数序列。")
        if (
            self.adaptive_holding_min_slots <= 0
            or self.adaptive_holding_max_slots < self.adaptive_holding_min_slots
        ):
            raise ValueError("adaptive holding slot 范围必须满足 1 <= min <= max。")
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
        self.macro_termination_scorer = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 2)
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

    def _macro_cache_logits(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """计算 Cache 候选 logits、critic value 与合法候选 mask。"""

        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observations, device)
        if int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Macro observation dimension mismatch。")
        features, masks = self._candidate_batch(
            cache_candidate_features,
            cache_candidate_masks,
            int(obs.shape[0]),
            self.macro_cache_feature_dim,
            device,
        )
        hidden = self.macro_actor_encoder(obs)
        candidate_hidden = self.macro_cache_candidate_encoder(features)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, int(features.shape[1]), -1)
        raw_logits = self.macro_cache_scorer(
            torch.cat((expanded_hidden, candidate_hidden), dim=-1)
        ).squeeze(-1)
        corrected_logits = self._cardinality_corrected_cache_logits(raw_logits, masks)
        values = self.macro_value(obs)
        return corrected_logits, raw_logits, values, masks

    def _macro_cache_node_logits(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Score node heads with shared parameters and the same global state."""

        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observations, device)
        if int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Macro observation dimension mismatch.")
        features = torch.as_tensor(
            cache_candidate_features, dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(cache_candidate_masks, dtype=torch.bool, device=device)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if (
            features.ndim != 4
            or tuple(features.shape[:1]) != (int(obs.shape[0]),)
            or tuple(features.shape[:-1]) != tuple(masks.shape)
            or int(features.shape[-1]) != self.macro_cache_feature_dim
            or not int(features.shape[1])
            or int(features.shape[2]) < 1
        ):
            raise ValueError(
                "factorized Cache tensors must be [B,N,K,D] with aligned [B,N,K] masks"
            )
        if not bool(torch.all(masks[..., 0]).item()) or not bool(
            torch.all(torch.any(masks, dim=-1)).item()
        ):
            raise ValueError("each factorized Cache head needs a legal KEEP at index zero")
        if not bool(torch.isfinite(features[masks]).all().item()):
            raise FloatingPointError("legal factorized Cache features contain NaN/Inf")
        hidden = self.macro_actor_encoder(obs)
        candidate_hidden = self.macro_cache_candidate_encoder(features)
        expanded_hidden = hidden[:, None, None, :].expand(
            -1, int(features.shape[1]), int(features.shape[2]), -1
        )
        raw_logits = self.macro_cache_scorer(
            torch.cat((expanded_hidden, candidate_hidden), dim=-1)
        ).squeeze(-1)
        corrected_logits = self._cardinality_corrected_cache_logits(raw_logits, masks)
        values = self.macro_value(obs)
        return corrected_logits, raw_logits, values, masks

    def sample_macro_nodes(
        self,
        observation: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        deterministic: bool = False,
    ) -> MultiNodeMacroActionSample:
        """Sample one cardinality-corrected categorical action per cache node."""

        if self.unified_cache_action:
            raise ValueError("sample_macro_nodes requires factorized Cache action semantics")
        device = next(self.parameters()).device
        features = torch.as_tensor(
            cache_candidate_features, dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(cache_candidate_masks, dtype=torch.bool, device=device)
        if features.ndim != 3 or masks.ndim != 2:
            raise ValueError("single-state factorized Cache inputs must be [N,K,D]/[N,K]")
        with torch.no_grad():
            obs, _ = self._observation_batch(observation, device)
            if int(obs.shape[0]) != 1:
                raise ValueError("sample_macro_nodes accepts one observation")
            logits, raw_logits, values, legal_masks = self._macro_cache_node_logits(
                obs, features, masks
            )
            node_logits = logits[0]
            node_raw_logits = raw_logits[0]
            node_masks = legal_masks[0]
            corrected = _masked_categorical(node_logits, node_masks)
            raw = _masked_categorical(node_raw_logits, node_masks)
            if not bool(torch.isfinite(corrected.probs).all().item()) or not torch.allclose(
                corrected.probs.sum(dim=-1),
                torch.ones(corrected.probs.shape[0], dtype=corrected.probs.dtype, device=device),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise FloatingPointError("factorized Cache probabilities are invalid")
            if deterministic:
                actions: list[int] = []
                diagnostics: list[dict[str, Any]] = []
                for node_index in range(int(node_logits.shape[0])):
                    group_ids = torch.ones_like(node_masks[node_index], dtype=torch.long)
                    group_ids[0] = 0
                    selected, selected_group = _group_aware_deterministic_candidate(
                        node_raw_logits[node_index],
                        corrected.probs[node_index],
                        node_masks[node_index],
                        group_ids,
                    )
                    update_mask = node_masks[node_index] & (group_ids == 1)
                    update_count = int(update_mask.sum().item())
                    diagnostics.append(
                        {
                            "node_index": node_index,
                            "selected_candidate_index": selected,
                            "selected_group": "KEEP" if selected_group == 0 else "UPDATE",
                            "p_keep": float(corrected.probs[node_index, 0].cpu()),
                            "p_update": float(corrected.probs[node_index, 1:].sum().cpu()),
                            "legal_update_candidate_count": update_count,
                            "selected_candidate_is_legal": bool(node_masks[node_index, selected].item()),
                        }
                    )
                    actions.append(selected)
                action_values = torch.as_tensor(actions, dtype=torch.long, device=device)
                deterministic_diagnostics = tuple(diagnostics)
            else:
                action_values = corrected.sample()
                deterministic_diagnostics = None
            node_log_probs = corrected.log_prob(action_values)
            node_entropies = corrected.entropy()
            return MultiNodeMacroActionSample(
                cache_actions=action_values.detach().cpu().numpy().astype(np.int64, copy=True),
                node_log_probs=node_log_probs,
                node_entropies=node_entropies,
                value=values.squeeze(0),
                cache_masks=node_masks.detach().cpu().numpy().copy(),
                node_probabilities=corrected.probs.detach().cpu().numpy().copy(),
                raw_node_probabilities=raw.probs.detach().cpu().numpy().copy(),
                node_p_keep=corrected.probs[:, 0],
                node_p_update=corrected.probs[:, 1:].sum(dim=-1),
                deterministic_diagnostics=deterministic_diagnostics,
            )

    def _sparse_cache_distributions(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        """Build P(group|s) and P(candidate|group,s) from legal update features."""

        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observations, device)
        features = torch.as_tensor(
            cache_candidate_features, dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(
            cache_candidate_masks, dtype=torch.bool, device=device
        )
        if features.ndim == 3:
            features = features.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if (
            features.ndim != 4
            or tuple(features.shape[:1]) != (int(obs.shape[0]),)
            or tuple(features.shape[:-1]) != tuple(masks.shape)
            or int(features.shape[-1]) != self.macro_cache_feature_dim
            or int(features.shape[1]) < 1
            or int(features.shape[2]) < 1
            or not bool(torch.all(masks[..., 0]).item())
        ):
            raise ValueError("sparse Cache tensors must be aligned [B,N,K,D]/[B,N,K]")
        if not bool(torch.isfinite(features[masks]).all().item()):
            raise FloatingPointError("legal sparse Cache features contain NaN/Inf")

        hidden = self.macro_actor_encoder(obs)
        candidate_hidden = self.macro_cache_candidate_encoder(features)
        batch_size, node_count, candidate_count = map(int, masks.shape)
        update_masks = masks[..., 1:]
        update_counts = update_masks.sum(dim=-1)
        has_update = update_counts > 0
        expanded_hidden = hidden[:, None, :].expand(-1, node_count, -1)

        if candidate_count > 1:
            update_embeddings = candidate_hidden[:, :, 1:, :]
            weights = update_masks.to(dtype=update_embeddings.dtype).unsqueeze(-1)
            pooled_update_embeddings = (update_embeddings * weights).sum(dim=2) / (
                update_counts.clamp_min(1).to(dtype=update_embeddings.dtype).unsqueeze(-1)
            )
            conditional_logits = self.macro_cache_scorer(
                torch.cat(
                    (
                        expanded_hidden.unsqueeze(2).expand(-1, -1, candidate_count - 1, -1),
                        update_embeddings,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            safe_update_masks = update_masks.clone()
            safe_update_masks[..., 0] |= ~has_update
            conditional_distribution = _masked_categorical(
                conditional_logits, safe_update_masks
            )
            conditional_probabilities = torch.where(
                update_masks,
                conditional_distribution.probs,
                torch.zeros_like(conditional_distribution.probs),
            )
            conditional_entropies = torch.where(
                has_update,
                conditional_distribution.entropy(),
                torch.zeros_like(update_counts, dtype=hidden.dtype),
            )
        else:
            pooled_update_embeddings = torch.zeros_like(expanded_hidden)
            conditional_distribution = None
            conditional_probabilities = features.new_zeros(
                (batch_size, node_count, 0)
            )
            conditional_entropies = features.new_zeros((batch_size, node_count))

        group_features = torch.cat(
            (
                expanded_hidden,
                pooled_update_embeddings,
                torch.log1p(update_counts.to(dtype=hidden.dtype)).unsqueeze(-1),
            ),
            dim=-1,
        )
        node_group_logits = self.macro_cache_group_scorer(group_features).squeeze(-1)
        keep_logits = self.macro_cache_keep_scorer(hidden)
        group_logits = torch.cat((keep_logits, node_group_logits), dim=-1)
        group_masks = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                has_update,
            ),
            dim=-1,
        )
        group_distribution = _masked_categorical(group_logits, group_masks)
        group_probabilities = group_distribution.probs
        if candidate_count > 1:
            node_update_probabilities = group_probabilities[:, 1:].unsqueeze(-1) * (
                conditional_probabilities
            )
        else:
            node_update_probabilities = conditional_probabilities
        node_keep_probabilities = 1.0 - group_probabilities[:, 1:]
        node_probabilities = torch.cat(
            (node_keep_probabilities.unsqueeze(-1), node_update_probabilities),
            dim=-1,
        )
        node_marginal_entropies = -torch.where(
            node_probabilities > 0,
            node_probabilities * torch.log(node_probabilities.clamp_min(1.0e-30)),
            torch.zeros_like(node_probabilities),
        ).sum(dim=-1)
        values = self.macro_value(obs)
        return (
            group_distribution,
            conditional_distribution,
            conditional_probabilities,
            conditional_entropies,
            values,
            masks,
            node_probabilities,
        )

    def sample_sparse_cache(
        self,
        observation: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        deterministic: bool = False,
    ) -> SparseCoordinatedCacheActionSample:
        """Sample one group first, then only that node's legal UPDATE target."""

        with torch.no_grad():
            obs, single = self._observation_batch(
                observation, next(self.parameters()).device
            )
            if not single or int(obs.shape[0]) != 1:
                raise ValueError("sample_sparse_cache accepts one observation")
            (
                group_distribution,
                conditional_distribution,
                conditional_probabilities,
                conditional_entropies,
                values,
                masks,
                node_probabilities,
            ) = self._sparse_cache_distributions(
                obs, cache_candidate_features, cache_candidate_masks
            )
            group_probs = group_distribution.probs[0]
            selected_group = 0
            selected_candidate = 0
            diagnostic = None
            if deterministic:
                if conditional_distribution is not None:
                    joint_update_probs = (
                        group_probs[1:, None]
                        * conditional_probabilities[0]
                    )
                    if joint_update_probs.numel():
                        flat_index = int(torch.argmax(joint_update_probs.reshape(-1)).item())
                        best_update_probability = float(
                            joint_update_probs.reshape(-1)[flat_index].item()
                        )
                        if best_update_probability > float(group_probs[0].item()):
                            candidate_count = int(masks.shape[-1]) - 1
                            selected_group = flat_index // candidate_count + 1
                            selected_candidate = flat_index % candidate_count + 1
                diagnostic = {
                    "selected_group": "ALL_KEEP" if selected_group == 0 else "UPDATE_NODE",
                    "selected_node_index": None if selected_group == 0 else selected_group - 1,
                    "selected_candidate_index": None if selected_group == 0 else selected_candidate,
                    "group_probabilities": group_probs.detach().cpu().numpy().tolist(),
                    "selected_joint_probability": float(
                        group_probs[selected_group].item()
                        if selected_group == 0
                        else group_probs[selected_group].item()
                        * conditional_probabilities[
                            0, selected_group - 1, selected_candidate - 1
                        ].item()
                    ),
                }
            else:
                selected_group = int(group_distribution.sample().item())
                if selected_group > 0:
                    assert conditional_distribution is not None
                    conditional_logits = conditional_distribution.logits[
                        0, selected_group - 1
                    ]
                    conditional_action = torch.distributions.Categorical(
                        logits=conditional_logits
                    ).sample()
                    selected_candidate = int(conditional_action.item()) + 1

            cache_actions = np.zeros(int(masks.shape[1]), dtype=np.int64)
            conditional_log_prob = group_probs.new_zeros(())
            if selected_group > 0:
                cache_actions[selected_group - 1] = selected_candidate
                assert conditional_distribution is not None
                conditional_actions = torch.zeros(
                    (1, int(masks.shape[1])), dtype=torch.long, device=obs.device
                )
                conditional_actions[0, selected_group - 1] = selected_candidate - 1
                conditional_log_prob = conditional_distribution.log_prob(
                    conditional_actions
                )[0, selected_group - 1]
            group_action = torch.as_tensor(selected_group, dtype=torch.long, device=obs.device)
            group_log_prob = group_distribution.log_prob(group_action)
            cache_log_prob = _sum_log_probabilities(
                torch.stack((group_log_prob.reshape(()), conditional_log_prob.reshape(()))),
                dim=0,
            )
            cache_entropy = group_distribution.entropy()[0] + torch.sum(
                group_probs[1:] * conditional_entropies[0]
            )
            row_probs = node_probabilities[0]
            tiny = torch.finfo(row_probs.dtype).tiny
            node_marginal_entropies = -torch.where(
                row_probs > 0,
                row_probs * torch.log(row_probs.clamp_min(tiny)),
                torch.zeros_like(row_probs),
            ).sum(dim=-1)
            return SparseCoordinatedCacheActionSample(
                cache_actions=cache_actions,
                group_action=selected_group,
                group_log_prob=group_log_prob,
                conditional_log_prob=conditional_log_prob,
                cache_log_prob=cache_log_prob,
                cache_entropy=cache_entropy,
                value=values.squeeze(0),
                group_probabilities=group_probs.detach().cpu().numpy().copy(),
                conditional_probabilities=conditional_probabilities[0].detach().cpu().numpy().copy(),
                node_probabilities=row_probs.detach().cpu().numpy().copy(),
                node_p_keep=1.0 - group_probs[1:],
                node_p_update=group_probs[1:],
                node_marginal_entropies=node_marginal_entropies,
                deterministic_diagnostics=diagnostic,
            )

    @staticmethod
    def _integer_elapsed_batch(
        elapsed_holding_slots: Any,
        batch_size: int,
        device: Any,
    ) -> Any:
        """检查已持有时隙是有限整数，并返回长整型批量张量。"""

        elapsed_float = torch.as_tensor(
            elapsed_holding_slots, dtype=torch.float32, device=device
        )
        if elapsed_float.ndim == 0 and batch_size == 1:
            elapsed_float = elapsed_float.unsqueeze(0)
        if tuple(elapsed_float.shape) != (batch_size,):
            raise ValueError("elapsed_holding_slots 必须是 [B] 或单样本标量。")
        if not bool(torch.all(torch.isfinite(elapsed_float)).item()) or not bool(
            torch.all(elapsed_float == torch.round(elapsed_float)).item()
        ):
            raise ValueError("elapsed_holding_slots 必须是有限整数。")
        return elapsed_float.to(dtype=torch.long)

    def _termination_logits(
        self,
        observations: Any,
        elapsed_holding_slots: Any,
    ) -> tuple[Any, Any, bool]:
        """根据状态和归一化持有时间计算 CONTINUE/TERMINATE logits。"""

        device = next(self.parameters()).device
        obs, single = self._observation_batch(observations, device)
        if int(obs.shape[-1]) != self.observation_dim:
            raise ValueError("Termination observation dimension mismatch。")
        elapsed = self._integer_elapsed_batch(
            elapsed_holding_slots, int(obs.shape[0]), device
        )
        eligible = (
            (elapsed >= self.adaptive_holding_min_slots)
            & (elapsed < self.adaptive_holding_max_slots)
        )
        if not bool(torch.all(eligible).item()):
            raise ValueError(
                "只允许在 min_slots <= elapsed_holding_slots < max_slots 时采样 termination。"
            )
        elapsed_fraction = (
            elapsed.to(dtype=obs.dtype) / float(self.adaptive_holding_max_slots)
        ).unsqueeze(-1)
        hidden = self.macro_actor_encoder(obs)
        logits = self.macro_termination_scorer(
            torch.cat((hidden, elapsed_fraction), dim=-1)
        )
        if not bool(torch.all(torch.isfinite(logits)).item()):
            raise FloatingPointError("Termination logits 包含 NaN/Inf。")
        return logits, elapsed, single

    def sample_macro(
        self,
        observation: Any,
        cache_candidate_features: Any,
        cache_candidate_mask: Any,
        deterministic: bool = False,
    ) -> MacroActionSample:
        """只采样统一 Cache 候选动作，并保留基数校正。"""

        features = np.asarray(cache_candidate_features, dtype=np.float32)
        cache_mask = np.asarray(cache_candidate_mask, dtype=bool).reshape(-1)
        if features.ndim != 2 or len(features) != len(cache_mask) or not len(cache_mask):
            raise ValueError("Macro cache candidates/mask must be non-empty and aligned。")
        if not bool(cache_mask[0]):
            raise ValueError("Macro candidate 0 must remain the legal KEEP action.")
        device = next(self.parameters()).device
        with torch.no_grad():
            obs, _ = self._observation_batch(observation, device)
            if int(obs.shape[0]) != 1:
                raise ValueError("sample_macro 每次只接受一个 observation。")
            cache_mask_t = torch.as_tensor(cache_mask, dtype=torch.bool, device=device)
            logits, raw_logits, values, _ = self._macro_cache_logits(
                obs,
                torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0),
                cache_mask_t.unsqueeze(0),
            )
            candidate_logits = logits.squeeze(0)
            raw_candidate_logits = raw_logits.squeeze(0)
            corrected_cache_dist = _masked_categorical(candidate_logits, cache_mask_t)
            raw_cache_dist = _masked_categorical(raw_candidate_logits, cache_mask_t)

            deterministic_cache_diagnostics = None
            group_ids = torch.ones_like(cache_mask_t, dtype=torch.long)
            group_ids[0] = 0
            if deterministic:
                cache_index, selected_cache_group = _group_aware_deterministic_candidate(
                    raw_candidate_logits,
                    corrected_cache_dist.probs,
                    cache_mask_t,
                    group_ids,
                )
                probability_sum_error = float(
                    torch.abs(corrected_cache_dist.probs.sum() - 1.0).detach().cpu()
                )
                update_mask = cache_mask_t & (group_ids == 1)
                legal_update_count = int(update_mask.sum().item())
                legal_update_best_score = (
                    float(
                        raw_candidate_logits.masked_fill(
                            ~update_mask, -float("inf")
                        ).max().cpu()
                    )
                    if legal_update_count
                    else None
                )
                selected_raw_score = float(
                    raw_candidate_logits[cache_index].detach().cpu()
                )
                selected_candidate_is_legal = bool(cache_mask_t[cache_index].item())
                update_inner_argmax_violation = bool(
                    selected_cache_group == 1
                    and (
                        cache_index <= 0
                        or not selected_candidate_is_legal
                        or selected_raw_score != legal_update_best_score
                    )
                )
                p_keep_value = float(corrected_cache_dist.probs[0].detach().cpu())
                p_update_group_value = float(
                    corrected_cache_dist.probs[1:].sum().detach().cpu()
                )
                deterministic_cache_diagnostics = {
                    "p_keep": p_keep_value,
                    "p_update_group": p_update_group_value,
                    "new_selected_group": "KEEP" if selected_cache_group == 0 else "UPDATE",
                    "new_selected_candidate_index": cache_index,
                    "group_tie_break": "KEEP",
                    "legal_update_candidate_count": legal_update_count,
                    "selected_candidate_is_legal": selected_candidate_is_legal,
                    "selected_raw_score": selected_raw_score,
                    "legal_update_best_raw_score": legal_update_best_score,
                    "update_inner_argmax_violation": update_inner_argmax_violation,
                    "probability_sum_error": probability_sum_error,
                }
            else:
                cache_index = int(corrected_cache_dist.sample().item())
                selected_cache_group = 0 if cache_index == 0 else 1

            cache_tensor = torch.as_tensor(cache_index, dtype=torch.long, device=device)
            cache_probabilities = corrected_cache_dist.probs
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
            cache_log_prob = corrected_cache_dist.log_prob(cache_tensor)
            cache_entropy = corrected_cache_dist.entropy()
            p_keep = cache_probabilities[0]
            p_update = cache_probabilities[1:].sum()
            return MacroActionSample(
                cache_index=cache_index,
                log_prob=cache_log_prob,
                entropy=cache_entropy,
                value=values.squeeze(0),
                cache_mask=cache_mask.copy(),
                cache_probabilities=cache_probabilities.detach().cpu().numpy().copy(),
                raw_cache_probabilities=raw_cache_dist.probs.detach().cpu().numpy().copy(),
                cache_entropy=cache_entropy,
                cache_selected_group="KEEP" if selected_cache_group == 0 else "UPDATE",
                p_keep=p_keep,
                p_update=p_update,
                deterministic_cache_diagnostics=deterministic_cache_diagnostics,
            )

    def sample_termination(
        self,
        observation: Any,
        elapsed_holding_slots: int,
        deterministic: bool = False,
    ) -> TerminationActionSample:
        """采样可学习的二元终止决策；平局时确定性选择 CONTINUE。"""

        with torch.no_grad():
            logits, _, single = self._termination_logits(
                observation, elapsed_holding_slots
            )
            if not single:
                raise ValueError("sample_termination 每次只接受一个 observation。")
            distribution = torch.distributions.Categorical(logits=logits)
            action = (
                int(torch.argmax(distribution.logits, dim=-1).item())
                if deterministic
                else int(distribution.sample().item())
            )
            action_t = torch.as_tensor(action, dtype=torch.long, device=logits.device)
            return TerminationActionSample(
                selected_index=action,
                log_prob=distribution.log_prob(action_t).squeeze(0),
                entropy=distribution.entropy().squeeze(0),
                probabilities=distribution.probs.squeeze(0).detach().cpu().numpy().copy(),
            )

    def evaluate_termination(
        self,
        observations: Any,
        elapsed_holding_slots: Any,
        actions: Any,
    ) -> TerminationEvaluation:
        """评估一批被采样的 termination 决策。"""

        logits, _, single = self._termination_logits(
            observations, elapsed_holding_slots
        )
        action_values = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
        if action_values.ndim == 0 and int(logits.shape[0]) == 1:
            action_values = action_values.unsqueeze(0)
        if tuple(action_values.shape) != (int(logits.shape[0]),):
            raise ValueError("termination actions 必须是 [B] 或单样本标量。")
        if bool(torch.any((action_values < 0) | (action_values > 1)).item()):
            raise ValueError("termination action 仅允许 0=CONTINUE 或 1=TERMINATE。")
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(action_values)
        entropy = distribution.entropy()
        probabilities = distribution.probs
        if single:
            return TerminationEvaluation(
                log_prob=log_prob.squeeze(0),
                entropy=entropy.squeeze(0),
                probabilities=probabilities.squeeze(0),
            )
        return TerminationEvaluation(
            log_prob=log_prob,
            entropy=entropy,
            probabilities=probabilities,
        )

    def evaluate_macro_components(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_actions: Any,
        termination_observations: Any,
        termination_elapsed_slots: Any,
        termination_actions: Any,
        termination_masks: Any,
    ) -> MacroComponentsEvaluation:
        """评估 Cache 动作与 mask 内 termination 决策，并按帧求和。"""

        device = next(self.parameters()).device
        obs, _ = self._observation_batch(observations, device)
        batch_size = int(obs.shape[0])
        raw_features = torch.as_tensor(cache_candidate_features, device=device)
        node_cache_log_probs = None
        node_cache_entropies = None
        node_cache_probabilities = None
        if raw_features.ndim == 4:
            if self.unified_cache_action:
                raise ValueError("factorized Cache rollout cannot be evaluated by a unified policy")
            cache_action_values = torch.as_tensor(
                cache_actions, dtype=torch.long, device=device
            )
            if bool(getattr(self, "sparse_coordinated_cache_action", False)):
                (
                    group_distribution,
                    conditional_distribution,
                    _conditional_probabilities,
                    conditional_entropies,
                    values,
                    masks,
                    node_cache_probabilities,
                ) = self._sparse_cache_distributions(
                    obs, cache_candidate_features, cache_candidate_masks
                )
                cache_probabilities = node_cache_probabilities
                expected_action_shape = (batch_size, int(masks.shape[1]))
                if tuple(cache_action_values.shape) != expected_action_shape:
                    raise ValueError(f"sparse cache_actions must be {expected_action_shape}")
                if bool(torch.any((cache_action_values < 0) | (cache_action_values >= masks.shape[2])).item()):
                    raise ValueError("sparse Cache action index out of range")
                selected_legal = masks.gather(2, cache_action_values.unsqueeze(-1)).squeeze(-1)
                if not bool(torch.all(selected_legal).item()):
                    raise ValueError("sparse Cache action is masked by its saved legal mask")
                selected_nodes = cache_action_values > 0
                update_counts = selected_nodes.sum(dim=1)
                if bool(torch.any(update_counts > 1).item()):
                    raise ValueError("sparse Cache action may update at most one node per frame")
                group_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
                selected_positions = torch.nonzero(selected_nodes, as_tuple=False)
                conditional_log_prob = torch.zeros(
                    batch_size, dtype=group_distribution.logits.dtype, device=device
                )
                if selected_positions.numel():
                    rows = selected_positions[:, 0]
                    nodes = selected_positions[:, 1]
                    candidate_indices = cache_action_values[rows, nodes]
                    group_actions[rows] = nodes + 1
                    if conditional_distribution is None:
                        raise ValueError("sparse UPDATE selected without a conditional candidate distribution")
                    conditional_masks = masks[rows, nodes, 1:]
                    conditional_indices = candidate_indices - 1
                    if not bool(
                        torch.all(
                            conditional_masks.gather(1, conditional_indices[:, None]).squeeze(1)
                        ).item()
                    ):
                        raise ValueError("sparse Cache UPDATE target is not legal")
                    conditional_actions = torch.zeros(
                        (batch_size, int(masks.shape[1])),
                        dtype=torch.long,
                        device=device,
                    )
                    conditional_actions[rows, nodes] = conditional_indices
                    all_conditional_log_probs = conditional_distribution.log_prob(
                        conditional_actions
                    )
                    conditional_log_prob = conditional_log_prob.index_put(
                        (rows,), all_conditional_log_probs[rows, nodes]
                    )
                group_log_prob = group_distribution.log_prob(group_actions)
                cache_log_prob = _sum_log_probabilities(
                    torch.stack((group_log_prob, conditional_log_prob), dim=-1),
                    dim=-1,
                )
                cache_entropy = group_distribution.entropy() + torch.sum(
                    group_distribution.probs[:, 1:] * conditional_entropies,
                    dim=-1,
                )
                p_keep = group_distribution.probs[:, 0]
                p_update = group_distribution.probs[:, 1:].sum(dim=-1)
            else:
                cache_logits, _, values, masks = self._macro_cache_node_logits(
                    obs,
                    cache_candidate_features,
                    cache_candidate_masks,
                )
                expected_action_shape = (batch_size, int(masks.shape[1]))
                if tuple(cache_action_values.shape) != expected_action_shape:
                    raise ValueError(f"factorized cache_actions must be {expected_action_shape}")
                if bool(torch.any((cache_action_values < 0) | (cache_action_values >= masks.shape[2])).item()):
                    raise ValueError("factorized Cache action index out of range")
                selected_legal = masks.gather(2, cache_action_values.unsqueeze(-1)).squeeze(-1)
                if not bool(torch.all(selected_legal).item()):
                    raise ValueError("factorized Cache action is masked by its saved legal mask")
                cache_distribution = _masked_categorical(cache_logits, masks)
                node_cache_log_probs = cache_distribution.log_prob(cache_action_values)
                node_cache_entropies = cache_distribution.entropy()
                node_cache_probabilities = cache_distribution.probs
                cache_log_prob = _sum_log_probabilities(node_cache_log_probs, dim=-1)
                cache_entropy = node_cache_entropies.sum(dim=-1)
                cache_probabilities = node_cache_probabilities
                p_keep = cache_probabilities[:, :, 0]
                p_update = cache_probabilities[:, :, 1:].sum(dim=-1)
        else:
            cache_logits, _, values, masks = self._macro_cache_logits(
                obs,
                cache_candidate_features,
                cache_candidate_masks,
            )
            cache_action_values = torch.as_tensor(
                cache_actions, dtype=torch.long, device=device
            ).reshape(-1)
            if tuple(cache_action_values.shape) != (batch_size,):
                raise ValueError("cache_actions 必须是 [B]。")
            if bool(torch.any((cache_action_values < 0) | (cache_action_values >= masks.shape[1])).item()):
                raise ValueError("Cache action index 越界。")
            if not bool(
                torch.all(masks.gather(1, cache_action_values[:, None]).squeeze(1)).item()
            ):
                raise ValueError("Cache action 被 rollout 保存的合法动作 mask 屏蔽。")
            cache_distribution = _masked_categorical(cache_logits, masks)
            cache_log_prob = cache_distribution.log_prob(cache_action_values)
            cache_entropy = cache_distribution.entropy()
            cache_probabilities = cache_distribution.probs
            p_keep = cache_probabilities[:, 0]
            p_update = cache_probabilities[:, 1:].sum(dim=-1)

        term_obs = torch.as_tensor(
            termination_observations, dtype=torch.float32, device=device
        )
        elapsed_slots = torch.as_tensor(termination_elapsed_slots, device=device)
        term_actions = torch.as_tensor(
            termination_actions, dtype=torch.long, device=device
        )
        term_masks = torch.as_tensor(
            termination_masks, dtype=torch.bool, device=device
        )
        if (
            term_obs.ndim != 3
            or tuple(term_obs.shape[:1]) != (batch_size,)
            or int(term_obs.shape[-1]) != self.observation_dim
        ):
            raise ValueError("termination_observations 必须是 [B,T,obs_dim]。")
        term_shape = tuple(term_obs.shape[:2])
        if (
            tuple(elapsed_slots.shape) != term_shape
            or tuple(term_actions.shape) != term_shape
            or tuple(term_masks.shape) != term_shape
        ):
            raise ValueError(
                "termination_elapsed_slots/actions/masks 必须都是 [B,T]。"
            )

        # 只选出 rollout 中真实采样的决策，padding 不进入 scorer 或概率和。
        selected_mask = term_masks.reshape(-1)
        flat_term_obs = term_obs.reshape(-1, self.observation_dim)[selected_mask]
        flat_elapsed = elapsed_slots.reshape(-1)[selected_mask]
        flat_actions = term_actions.reshape(-1)[selected_mask]
        if flat_term_obs.shape[0]:
            evaluated = self.evaluate_termination(
                flat_term_obs, flat_elapsed, flat_actions
            )
            selected_log_probs = evaluated.log_prob.reshape(-1)
            selected_entropies = evaluated.entropy.reshape(-1)
            selected_probabilities = evaluated.probabilities.reshape(-1, 2)
            term_log_probs = torch.zeros(
                term_shape, dtype=selected_log_probs.dtype, device=device
            ).masked_scatter(term_masks, selected_log_probs)
            term_entropies = torch.zeros(
                term_shape, dtype=selected_entropies.dtype, device=device
            ).masked_scatter(term_masks, selected_entropies)
            expanded_masks = term_masks.unsqueeze(-1).expand(-1, -1, 2)
            term_probabilities = torch.zeros(
                (*term_shape, 2),
                dtype=selected_probabilities.dtype,
                device=device,
            ).masked_scatter(expanded_masks, selected_probabilities.reshape(-1))
        else:
            term_log_probs = torch.zeros(term_shape, dtype=cache_log_prob.dtype, device=device)
            term_entropies = torch.zeros(term_shape, dtype=cache_entropy.dtype, device=device)
            term_probabilities = torch.zeros(
                (*term_shape, 2), dtype=cache_probabilities.dtype, device=device
            )

        termination_log_prob_sum, joint_log_prob = _macro_log_prob_from_components(
            cache_log_prob, term_log_probs, dim=1
        )
        termination_entropy_sum = term_entropies.sum(dim=1)
        decomposition_error = torch.abs(
            joint_log_prob.to(torch.float64)
            - cache_log_prob.to(torch.float64)
            - termination_log_prob_sum.to(torch.float64)
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
            termination_log_probs=term_log_probs,
            termination_entropies=term_entropies,
            termination_probabilities=term_probabilities,
            termination_log_prob_sum=termination_log_prob_sum,
            termination_entropy_sum=termination_entropy_sum,
            p_keep=p_keep,
            p_update=p_update,
            joint_log_prob=joint_log_prob,
            cache_entropy=cache_entropy,
            value=values,
            joint_logprob_decomposition_error=decomposition_error,
            node_cache_log_probs=node_cache_log_probs,
            node_cache_entropies=node_cache_entropies,
            node_cache_probabilities=node_cache_probabilities,
        )

    def evaluate_macro_joint(
        self,
        observations: Any,
        cache_candidate_features: Any,
        cache_candidate_masks: Any,
        cache_actions: Any,
        termination_observations: Any,
        termination_elapsed_slots: Any,
        termination_actions: Any,
        termination_masks: Any,
    ) -> Any:
        """返回 Cache 与所有已采样 termination 决策的联合 log-prob。"""

        return self.evaluate_macro_components(
            observations,
            cache_candidate_features,
            cache_candidate_masks,
            cache_actions,
            termination_observations,
            termination_elapsed_slots,
            termination_actions,
            termination_masks,
        ).joint_log_prob

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
        head_group_stay_probabilities: list[Any] = []
        head_group_move_probabilities: list[Any] = []
        head_group_entropies: list[Any] = []
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
            # 这里沿用 PPO 概率计算使用的 masked、group-balanced 分布。
            # candidate 0 是 STAY，其余候选的概率质量都属于 MOVE。
            # 保留梯度连接，使训练器只通过 Micro Actor 更新该正则项。
            p_stay = dist.probs[:, 0]
            p_move = dist.probs[:, 1:].sum(dim=-1)
            tiny = torch.finfo(dist.probs.dtype).tiny
            group_entropy = -(
                torch.where(
                    p_stay > 0,
                    p_stay * torch.log(p_stay.clamp_min(tiny)),
                    torch.zeros_like(p_stay),
                )
                + torch.where(
                    p_move > 0,
                    p_move * torch.log(p_move.clamp_min(tiny)),
                    torch.zeros_like(p_move),
                )
            )
            head_group_stay_probabilities.append(p_stay)
            head_group_move_probabilities.append(p_move)
            head_group_entropies.append(group_entropy)
        values = self.micro_critic(self._micro_critic_inputs(obs)).squeeze(-1)
        log0, log1 = head_log_probs
        entropies = head_entropies[0] + head_entropies[1]
        group_entropy0, group_entropy1 = head_group_entropies
        joint_group_entropy = group_entropy0 + group_entropy1
        return JointMicroEvaluation(
            log_prob=log0 + log1,
            entropy=entropies,
            value=values,
            uav0_log_prob=log0,
            uav1_log_prob=log1,
            uav0_stay_probability=head_group_stay_probabilities[0],
            uav0_move_probability=head_group_move_probabilities[0],
            uav0_group_entropy=group_entropy0,
            uav1_stay_probability=head_group_stay_probabilities[1],
            uav1_move_probability=head_group_move_probabilities[1],
            uav1_group_entropy=group_entropy1,
            joint_group_entropy=joint_group_entropy,
            mean_group_entropy=joint_group_entropy / 2.0,
        )

    def macro_actor_parameters(self) -> tuple[Any, ...]:
        modules = (
            self.macro_actor_encoder,
            self.macro_cache_candidate_encoder,
            self.macro_cache_scorer,
            self.macro_termination_scorer,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

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
            "macro_termination_scorer.",
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
    ) -> None:
        long_state = {
            f"long_value_critic.{key}": value
            for key, value in long_value_critic_state.items()
        }
        combined = {**macro_state, **micro_state, **long_state}
        self.load_state_dict(combined, strict=True)


class MultiNodeFactorizedActorCritic(DualTimescaleActorCritic):
    """Versioned nine-head Cache policy sharing the established actor/scorer."""

    architecture_id = (
        "adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_"
        "v4_multi_node_cache_factorized_v1"
    )
    architecture_version = 33

    def __init__(
        self,
        observation_dim: int,
        macro_cache_feature_dim: int,
        micro_uav_feature_dim: int,
        adaptive_holding_min_slots: int,
        adaptive_holding_max_slots: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            macro_cache_feature_dim,
            micro_uav_feature_dim,
            adaptive_holding_min_slots,
            adaptive_holding_max_slots,
            hidden_sizes,
            unified_cache_action=False,
            cardinality_corrected_cache_action=True,
        )


class SparseCoordinatedCacheActorCritic(MultiNodeFactorizedActorCritic):
    """Versioned hierarchical Cache actor allowing at most one node UPDATE."""

    architecture_id = (
        "adaptive_dual_timescale_proposed_cache_economic_features_fixed_lr_horizon_mask_"
        "v5_sparse_coordinated_cache_v1"
    )
    architecture_version = 34
    sparse_coordinated_cache_action = True

    def __init__(
        self,
        observation_dim: int,
        macro_cache_feature_dim: int,
        micro_uav_feature_dim: int,
        adaptive_holding_min_slots: int,
        adaptive_holding_max_slots: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            observation_dim,
            macro_cache_feature_dim,
            micro_uav_feature_dim,
            adaptive_holding_min_slots,
            adaptive_holding_max_slots,
            hidden_sizes,
        )
        self.cardinality_corrected_cache_action = False
        hidden = int(tuple(hidden_sizes)[-1])
        self.macro_cache_keep_scorer = nn.Linear(hidden, 1)
        self.macro_cache_group_scorer = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def macro_actor_parameters(self) -> tuple[Any, ...]:
        modules = (
            self.macro_actor_encoder,
            self.macro_cache_candidate_encoder,
            self.macro_cache_scorer,
            self.macro_cache_keep_scorer,
            self.macro_cache_group_scorer,
            self.macro_termination_scorer,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

    def macro_state_dict(self) -> dict[str, Any]:
        prefixes = (
            "macro_actor_encoder.",
            "macro_cache_candidate_encoder.",
            "macro_cache_scorer.",
            "macro_cache_keep_scorer.",
            "macro_cache_group_scorer.",
            "macro_termination_scorer.",
            "macro_critic.",
        )
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }
