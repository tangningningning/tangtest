"""Stage-1 standard MAPPO path reusing the formal Cache/UAV actor networks.

The collector keeps the formal prediction and Matching+KKT environment path.
Cache decisions occur at a fixed two-slot cadence; UAV decisions occur every
slot. PPO ratios use each cooperative group's joint action log-probability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]

from common import project_config
from decision import cache_configuration, slot_environment, uav_configuration

from .policy import MultiNodeFactorizedActorCritic, _masked_categorical
from .cache_counterfactual_value import (
    COUNTERFACTUAL_VALUE_SCHEMA,
    CACHE_NODE_VALUE_FEATURE_DIM,
    CacheDeltaQValueHead,
    CounterfactualMacroTrace,
    Stage3CacheValueConfig,
    cache_node_value_features,
    candidate_value_pairwise_ranking_loss,
    delta_q_huber_loss,
    evaluate_counterfactual_trace,
    make_counterfactual_trace,
    conditional_policy_expected_delta_q,
    policy_state_fingerprint,
    trigger_value_ranking_loss,
)
from .primitive_control import (
    CACHE_ECONOMIC_BASE_MACRO_FEATURE_DIM,
    CACHE_ECONOMIC_FEATURE_DIM,
    PrimitiveTensorSpec,
)
from .rollout import (
    _cache_node_candidates,
    _check_request_contract,
    _check_reward,
    _observation,
    _pad_micro_candidates,
    _positions_condition,
    _uav_candidates,
    compute_gae,
    compute_macro_smdp_gae,
)
from .ppo_math import _clipped_ppo_surrogate_loss, _normalize_advantages


MACRO_INTERVAL_SLOTS = 2
MAPPO_BASELINE_SCHEMA = (
    "mappo_shared_actor_global_critic_fixed_macro_2_v2_group_balanced_ops"
)
MAPPO_CHECKPOINT_VERSION = 2
MAPPO_STAGE3_LEGACY_CHECKPOINT_VERSION = 4
MAPPO_STAGE3_CHECKPOINT_VERSION = 5
MACRO_CONDITIONAL_CANDIDATE_PREFIXES = (
    "macro_conditional_candidate_encoder.",
    "macro_conditional_cache_scorer.",
)


class CounterfactualLabelBatchError(RuntimeError):
    """Stage3 batch failure that carries the rejected-label diagnostics."""

    def __init__(self, message: str, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


@dataclass(frozen=True)
class MAPPOMacroTransition:
    """One Cache decision and its fixed-duration SMDP return segment."""

    slot: int
    observation: np.ndarray
    candidate_features: np.ndarray
    candidate_masks: np.ndarray
    actions: np.ndarray
    old_trigger_log_probs: np.ndarray
    old_content_log_probs: np.ndarray
    old_node_log_probs: np.ndarray
    old_joint_log_prob: float
    old_value: float
    executed_cache_configuration: np.ndarray
    frame_reward: float
    next_observation: np.ndarray
    old_next_value: float
    duration: int
    done: bool
    actual_cache_update_by_node: np.ndarray
    actor_inference_seconds: float


@dataclass(frozen=True)
class MAPPOMicroTransition:
    """One slot's sequential two-UAV action and legal replay support."""

    slot: int
    observation: np.ndarray
    candidate_features: np.ndarray
    candidate_masks: np.ndarray
    candidate_positions: np.ndarray
    actions: np.ndarray
    uav0_position_condition: np.ndarray
    executed_target_positions: np.ndarray
    old_head_log_probs: np.ndarray
    old_joint_log_prob: float
    old_value: float
    reward: float
    next_observation: np.ndarray
    done: bool
    actual_uav_moves: np.ndarray
    matching_summary: dict[str, Any]
    actual_task_utility: float
    cache_cost: float
    movement_cost: float
    movement_energy_j: float
    matching_kkt_seconds: float
    actor_inference_seconds: float


@dataclass(frozen=True)
class MAPPORollout:
    """Short or complete episode data collected by a frozen behavior policy."""

    macro: tuple[MAPPOMacroTransition, ...]
    micro: tuple[MAPPOMicroTransition, ...]
    summary: dict[str, Any]
    counterfactual_traces: tuple[CounterfactualMacroTrace, ...] = ()


class MAPPOBaselineTrainer:
    """Fixed-cadence MAPPO using the existing factorized nine-node/two-UAV model."""

    def __init__(
        self,
        context: Any,
        *,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("Stage-1 MAPPO requires PyTorch.")
        self.context = context
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        if (
            context.cache_action_mode
            != project_config.CACHE_ACTION_MODE_MULTI_NODE_FACTORIZED
        ):
            raise ValueError("Stage-1 MAPPO requires nine factorized Cache heads.")
        if len(context.action_spec.cache_node_ids) != 9:
            raise ValueError("Stage-1 MAPPO requires exactly nine Cache nodes.")
        if len(context.action_spec.uav_ids) != 2:
            raise ValueError("Stage-1 MAPPO requires exactly two UAVs.")
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            random.seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        self.seed = None if seed is None else int(seed)
        self.update_count = 0
        self.generator = torch.Generator(device=self.device)
        if self.seed is not None:
            self.generator.manual_seed(self.seed + 1)
        self.counterfactual_micro_rng: random.Random | None = None

        ppo = context.cfg["algorithm"]["ppo"]
        self.gamma = float(ppo["gamma"])
        self.gae_lambda = float(ppo["gae_lambda"])
        self.clip_epsilon = float(ppo["clip_epsilon"])
        self.max_grad_norm = float(ppo["max_grad_norm"])
        if not all(
            math.isfinite(value)
            for value in (self.gamma, self.gae_lambda, self.clip_epsilon, self.max_grad_norm)
        ):
            raise ValueError("MAPPO hyperparameters must be finite.")
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gamma and gae_lambda must be in [0, 1].")
        if self.clip_epsilon <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("clip_epsilon and max_grad_norm must be positive.")

        tensor_spec = PrimitiveTensorSpec.create(context)
        macro_feature_dim = (
            CACHE_ECONOMIC_BASE_MACRO_FEATURE_DIM + CACHE_ECONOMIC_FEATURE_DIM
        )
        if macro_feature_dim != 200:
            raise ValueError("The reused formal multi-node Cache feature width changed.")
        self.policy = MultiNodeFactorizedActorCritic(
            int(context.observation_spec.observation_dim),
            macro_feature_dim,
            int(tensor_spec.uav_feature_dim) + 2,
            MACRO_INTERVAL_SLOTS,
            MACRO_INTERVAL_SLOTS,
            tuple(int(value) for value in ppo["hidden_sizes"]),
        ).to(self.device)
        for name, parameter in self.policy.named_parameters():
            if name.startswith(("macro_termination_scorer.", "long_value_critic.")):
                parameter.requires_grad_(False)
        self.tensor_spec = tensor_spec
        self.max_uav_candidates = int(tensor_spec.uav_max_candidates) + 1
        self.uav_feature_dim = int(tensor_spec.uav_feature_dim) + 2

        named_parameters = dict(self.policy.named_parameters())
        self.macro_actor_parameters = tuple(
            value
            for name, value in named_parameters.items()
            if name.startswith(
                (
                    "macro_actor_encoder.",
                    "macro_cache_candidate_encoder.",
                    "macro_cache_scorer.",
                    *MACRO_CONDITIONAL_CANDIDATE_PREFIXES,
                )
            )
        )
        self.macro_candidate_parameters = tuple(
            value
            for name, value in named_parameters.items()
            if name.startswith(MACRO_CONDITIONAL_CANDIDATE_PREFIXES)
        )
        self.micro_actor_parameters = tuple(
            value
            for name, value in named_parameters.items()
            if name.startswith(("micro_actor_encoder.", "micro_uav_candidate_scorer."))
        )
        self.macro_critic_parameters = tuple(
            value
            for name, value in named_parameters.items()
            if name.startswith("macro_critic.")
        )
        self.micro_critic_parameters = tuple(
            value
            for name, value in named_parameters.items()
            if name.startswith("micro_critic.")
        )
        if not all(
            (self.macro_actor_parameters, self.micro_actor_parameters,
             self.macro_critic_parameters, self.micro_critic_parameters)
        ):
            raise RuntimeError("Reused MAPPO actor/critic parameter groups are incomplete.")

        actor_lr = float(ppo.get("actor_learning_rate", ppo["learning_rate"]))
        macro_actor_lr = float(ppo.get("macro_actor_learning_rate", actor_lr))
        self.macro_actor_optimizer = torch.optim.Adam(
            self.macro_actor_parameters, lr=macro_actor_lr
        )
        self.micro_actor_optimizer = torch.optim.Adam(
            self.micro_actor_parameters, lr=actor_lr
        )
        self.macro_critic_optimizer = torch.optim.Adam(
            self.macro_critic_parameters,
            lr=float(ppo.get("macro_critic_learning_rate", ppo["critic_learning_rate"])),
        )
        self.micro_critic_optimizer = torch.optim.Adam(
            self.micro_critic_parameters,
            lr=float(ppo.get("micro_critic_learning_rate", ppo["critic_learning_rate"])),
        )
        self.stage3_cache_value_config: Stage3CacheValueConfig | None = None
        self.cache_delta_q_value_head: CacheDeltaQValueHead | None = None
        self.cache_delta_q_value_optimizer: Any | None = None
        self.stage3_update_count = 0
        self.candidate_supervised_update_count = 0
        self.counterfactual_rollout_count = 0
        self.stage3_initialization_mode: str | None = None
        self.stage3_source_checkpoint_sha256: str | None = None

    def enable_stage3_counterfactual_cache_value(
        self, config: Stage3CacheValueConfig
    ) -> None:
        """Enable the isolated Stage3 auxiliary head and bounded replay mode."""

        if self.stage3_cache_value_config is not None:
            raise RuntimeError("Stage3 counterfactual cache-value mode is already enabled.")
        config.validate()
        if self.policy.macro_cache_feature_dim != 200:
            raise ValueError("Stage3 currently requires the Stage1D 200-D Macro candidate schema.")
        if self.seed is None:
            raise ValueError(
                "Stage3 requires a trainer seed for its independent Micro common-random stream."
            )
        if not self.policy.separate_trigger_candidate:
            old_macro_optimizer_state = self.macro_actor_optimizer.state_dict()
            old_macro_optimizer_defaults = copy.deepcopy(
                self.macro_actor_optimizer.defaults
            )
            self.policy.enable_separate_trigger_candidate()
            named_parameters = dict(self.policy.named_parameters())
            self.macro_actor_parameters = tuple(
                parameter
                for name, parameter in named_parameters.items()
                if name.startswith(
                    (
                        "macro_actor_encoder.",
                        "macro_cache_candidate_encoder.",
                        "macro_cache_scorer.",
                        *MACRO_CONDITIONAL_CANDIDATE_PREFIXES,
                    )
                )
            )
            self.macro_candidate_parameters = tuple(
                parameter
                for name, parameter in named_parameters.items()
                if name.startswith(MACRO_CONDITIONAL_CANDIDATE_PREFIXES)
            )
            self.macro_actor_optimizer = torch.optim.Adam(
                self.macro_actor_parameters, **old_macro_optimizer_defaults
            )
            migrated_macro_optimizer_state = (
                self._migrate_macro_actor_optimizer_state(
                    old_macro_optimizer_state
                )
            )
            self._validate_optimizer_checkpoint_state(
                self.macro_actor_optimizer,
                migrated_macro_optimizer_state,
                optimizer_name="macro_actor_stage3_head_installation",
            )
            self.macro_actor_optimizer.load_state_dict(
                migrated_macro_optimizer_state
            )
        self.stage3_cache_value_config = config
        self.counterfactual_micro_rng = random.Random(
            (int(self.seed) ^ 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF
        )
        self.cache_delta_q_value_head = CacheDeltaQValueHead(
            int(self.context.observation_spec.observation_dim),
            int(self.policy.macro_cache_feature_dim),
            CACHE_NODE_VALUE_FEATURE_DIM,
            tuple(int(value) for value in self.context.cfg["algorithm"]["ppo"]["hidden_sizes"]),
        ).to(self.device)
        self.cache_delta_q_value_optimizer = torch.optim.Adam(
            self.cache_delta_q_value_head.parameters(),
            lr=float(config.value_head_learning_rate),
        )
        self.stage3_initialization_mode = "fresh_stage3_delta_q_head"
        self.stage3_source_checkpoint_sha256 = None

    def _selected_counterfactual_macro_indices(self, macro_count: int) -> tuple[int, ...]:
        if self.stage3_cache_value_config is None:
            return ()
        budget = int(self.stage3_cache_value_config.max_macro_states_per_rollout)
        selected_count = min(int(macro_count), budget)
        if selected_count <= 0:
            return ()
        return tuple(
            sorted({(index * int(macro_count)) // selected_count for index in range(selected_count)})
        )

    def _counterfactual_target_nodes(self, macro_index: int) -> tuple[int, ...]:
        if self.stage3_cache_value_config is None:
            return ()
        node_count = len(self.context.action_spec.cache_node_ids)
        selected_count = min(
            node_count,
            int(self.stage3_cache_value_config.max_target_nodes_per_state),
        )
        start = (
            self.counterfactual_rollout_count * selected_count
            + int(macro_index) * selected_count
        ) % node_count
        return tuple((start + offset) % node_count for offset in range(selected_count))

    def _macro_group_distribution(self, logits: Any, mask: Any) -> tuple[Any, Any]:
        """Factor corrected KEEP/UPDATE mass from legal candidate scores."""

        if logits.ndim != 1 or mask.ndim != 1 or tuple(logits.shape) != tuple(mask.shape):
            raise ValueError("A node Cache head must have aligned one-dimensional logits/mask.")
        if len(mask) < 1 or not bool(mask[0].item()):
            raise ValueError("Every Cache node must retain legal KEEP at index zero.")
        update_mask = mask[1:]
        has_update = bool(torch.any(update_mask).item())
        update_logit = (
            torch.logsumexp(logits[1:].masked_fill(~update_mask, -float("inf")), dim=0)
            if has_update
            else logits.new_tensor(-float("inf"))
        )
        group_logits = torch.stack((logits[0], update_logit))
        group_mask = torch.as_tensor(
            (True, has_update), dtype=torch.bool, device=logits.device
        )
        return _masked_categorical(group_logits, group_mask), update_mask

    def _micro_group_distribution(
        self, logits: Any, candidate_features: Any, mask: Any
    ) -> tuple[Any, tuple[Any, Any]]:
        """Factor STAY/MOVE mass from group-balanced legal candidate scores."""

        features = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=logits.device
        )
        legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if (
            logits.ndim != 1
            or legal.ndim != 1
            or tuple(features.shape) != (len(legal), self.uav_feature_dim)
            or tuple(logits.shape) != tuple(legal.shape)
            or not bool(torch.any(legal).item())
        ):
            raise ValueError("A Micro head needs aligned legal candidate scores/features.")
        group_markers = features[:, -2:]
        legal_markers = group_markers[legal]
        if (
            not bool(torch.all((legal_markers == 0.0) | (legal_markers == 1.0)).item())
            or not bool(torch.all(legal_markers.sum(dim=-1) == 1.0).item())
        ):
            raise ValueError("Every legal Micro candidate must belong to STAY or MOVE.")
        group_ids = group_markers.argmax(dim=-1)
        group_masks = tuple(legal & (group_ids == group) for group in range(2))
        group_mask = torch.as_tensor(
            [bool(torch.any(group_candidates).item()) for group_candidates in group_masks],
            dtype=torch.bool,
            device=logits.device,
        )
        group_logits = torch.stack(
            [
                torch.logsumexp(
                    logits.masked_fill(~group_candidates, -float("inf")), dim=0
                )
                if bool(group_mask[group].item())
                else logits.new_tensor(-float("inf"))
                for group, group_candidates in enumerate(group_masks)
            ]
        )
        return _masked_categorical(group_logits, group_mask), group_masks

    def _sample_macro_action(
        self,
        observation: np.ndarray,
        candidate_features: np.ndarray,
        candidate_masks: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Sample KEEP/UPDATE first, then a legal content only on UPDATE."""

        inference_started = time.perf_counter()
        with torch.no_grad():
            (
                trigger_logits,
                _,
                conditional_logits,
                _,
                _,
                masks,
            ) = self.policy._macro_cache_factorized_node_logits(
                observation, candidate_features, candidate_masks
            )
            node_trigger_logits = trigger_logits[0]
            node_conditional_logits = conditional_logits[0]
            node_masks = masks[0]
            if int(node_trigger_logits.shape[0]) != 9:
                raise ValueError("Macro Cache head count changed from nine.")
            conditional_distributions: list[Any | None] | None = None
            conditional_candidate_probabilities: np.ndarray | None = None
            if self.stage3_cache_value_config is not None:
                conditional_distributions = []
                conditional_candidate_probabilities = np.zeros(
                    tuple(node_masks.shape), dtype=np.float32
                )
                for node_index in range(int(node_masks.shape[0])):
                    update_mask = node_masks[node_index, 1:]
                    if bool(torch.any(update_mask).item()):
                        distribution = _masked_categorical(
                            node_conditional_logits[node_index, 1:], update_mask
                        )
                        conditional_distributions.append(distribution)
                        conditional_candidate_probabilities[node_index, 1:] = (
                            distribution.probs.detach().cpu().numpy().astype(
                                np.float32, copy=True
                            )
                        )
                    else:
                        conditional_distributions.append(None)
            actions: list[int] = []
            trigger_log_probs: list[Any] = []
            content_log_probs: list[Any] = []
            for node_index in range(9):
                trigger_distribution, update_mask = self._macro_group_distribution(
                    node_trigger_logits[node_index], node_masks[node_index]
                )
                trigger = (
                    torch.argmax(trigger_distribution.probs)
                    if deterministic
                    else torch.multinomial(
                        trigger_distribution.probs, 1, generator=self.generator
                    ).squeeze(0)
                )
                trigger_log_prob = trigger_distribution.log_prob(trigger)
                if int(trigger.item()) == 0:
                    selected = 0
                    content_log_prob = trigger_log_prob.new_zeros(())
                else:
                    conditional_distribution = (
                        _masked_categorical(
                            node_conditional_logits[node_index, 1:], update_mask
                        )
                        if conditional_distributions is None
                        else conditional_distributions[node_index]
                    )
                    if conditional_distribution is None:
                        raise RuntimeError(
                            "Macro UPDATE was sampled for a node without legal UPDATE candidates."
                        )
                    content = (
                        torch.argmax(conditional_distribution.probs)
                        if deterministic
                        else torch.multinomial(
                            conditional_distribution.probs, 1, generator=self.generator
                        ).squeeze(0)
                    )
                    selected = int(content.item()) + 1
                    content_log_prob = conditional_distribution.log_prob(content)
                actions.append(selected)
                trigger_log_probs.append(trigger_log_prob)
                content_log_probs.append(content_log_prob)
            action_values = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            trigger_log_values = torch.stack(trigger_log_probs)
            content_log_values = torch.stack(content_log_probs)
            node_log_values = trigger_log_values + content_log_values
            inference_seconds = time.perf_counter() - inference_started
            value = self.policy.macro_value(observation)
        return {
            "actions": action_values.cpu().numpy().copy(),
            "trigger_log_probs": trigger_log_values.cpu().numpy().astype(np.float32),
            "content_log_probs": content_log_values.cpu().numpy().astype(np.float32),
            "node_log_probs": node_log_values.cpu().numpy().astype(np.float32),
            "joint_log_prob": float(node_log_values.to(torch.float64).sum().cpu()),
            "value": float(value.cpu()),
            "masks": node_masks.cpu().numpy().copy(),
            "conditional_candidate_probabilities": (
                None
                if conditional_candidate_probabilities is None
                else conditional_candidate_probabilities
            ),
            "actor_inference_seconds": float(inference_seconds),
        }

    def _evaluate_macro_actions(
        self,
        observations: Any,
        candidate_features: Any,
        candidate_masks: Any,
        actions: Any,
    ) -> tuple[Any, Any, Any, Any]:
        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        features_t = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=self.device
        )
        masks_t = torch.as_tensor(
            candidate_masks, dtype=torch.bool, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        (
            trigger_logits,
            _,
            conditional_logits,
            _,
            values,
            legal_masks,
        ) = self.policy._macro_cache_factorized_node_logits(
            observations_t, features_t, masks_t
        )
        if tuple(actions_t.shape) != tuple(legal_masks.shape[:2]):
            raise ValueError("Macro actions must align with [batch, nine nodes].")
        if bool(torch.any((actions_t < 0) | (actions_t >= legal_masks.shape[-1])).item()):
            raise ValueError("Macro action index is outside its saved legal mask.")
        selected_legal = legal_masks.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.all(selected_legal).item()):
            raise ValueError("Macro action is masked by its behavior-time legal support.")

        batch_node_log_probs: list[Any] = []
        batch_entropies: list[Any] = []
        for batch_index in range(int(actions_t.shape[0])):
            node_log_probs: list[Any] = []
            node_entropies: list[Any] = []
            for node_index in range(9):
                node_mask = legal_masks[batch_index, node_index]
                node_trigger_logits = trigger_logits[batch_index, node_index]
                node_conditional_logits = conditional_logits[batch_index, node_index]
                action = int(actions_t[batch_index, node_index].item())
                trigger_distribution, update_mask = self._macro_group_distribution(
                    node_trigger_logits, node_mask
                )
                trigger = torch.as_tensor(
                    0 if action == 0 else 1,
                    dtype=torch.long,
                    device=self.device,
                )
                joint_log_prob = trigger_distribution.log_prob(trigger)
                conditional_distribution = None
                node_entropy = trigger_distribution.entropy()
                if bool(torch.any(update_mask).item()):
                    conditional_distribution = _masked_categorical(
                        node_conditional_logits[1:], update_mask
                    )
                    node_entropy = node_entropy + (
                        trigger_distribution.probs[1]
                        * conditional_distribution.entropy()
                    )
                if action > 0:
                    if conditional_distribution is None:
                        raise ValueError("Macro UPDATE selected without a legal candidate.")
                    content = torch.as_tensor(
                        action - 1, dtype=torch.long, device=self.device
                    )
                    joint_log_prob = joint_log_prob + conditional_distribution.log_prob(
                        content
                    )
                node_log_probs.append(joint_log_prob)
                node_entropies.append(node_entropy)
            batch_node_log_probs.append(torch.stack(node_log_probs))
            batch_entropies.append(torch.stack(node_entropies).sum())
        per_node_log_probs = torch.stack(batch_node_log_probs).to(dtype=torch.float32)
        joint_log_probs = per_node_log_probs.to(torch.float64).sum(dim=1).to(dtype=torch.float32)
        entropies = torch.stack(batch_entropies).to(dtype=torch.float32)
        return joint_log_probs, values, per_node_log_probs, entropies

    def _sample_micro_head(
        self,
        observation: np.ndarray,
        features: np.ndarray,
        mask: np.ndarray,
        *,
        uav_index: int,
        uav0_position_condition: np.ndarray,
        deterministic: bool = False,
        random_uniforms: np.ndarray | tuple[float, float] | None = None,
    ) -> tuple[int, float, np.ndarray, float]:
        if int(uav_index) not in (0, 1):
            raise ValueError("Micro policy requires the ordered UAV0/UAV1 heads.")
        inference_started = time.perf_counter()
        with torch.no_grad():
            logits = self.policy._micro_candidate_logits(
                observation, features, mask, uav0_position_condition
            )
            features_t = torch.as_tensor(
                features, dtype=torch.float32, device=self.device
            )
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            operation_distribution, operation_candidate_masks = (
                self._micro_group_distribution(logits, features_t, mask_t)
            )
            if random_uniforms is not None:
                if deterministic:
                    raise ValueError("Explicit Micro uniforms cannot be combined with deterministic sampling.")
                uniforms = np.asarray(random_uniforms, dtype=np.float64)
                if (
                    uniforms.shape != (2,)
                    or not np.isfinite(uniforms).all()
                    or np.any(uniforms < 0.0)
                    or np.any(uniforms >= 1.0)
                ):
                    raise ValueError("Micro common uniforms must be two finite values in [0, 1).")
                operation = self._sample_categorical_with_uniform(
                    operation_distribution.probs, float(uniforms[0])
                )
            elif deterministic:
                operation = torch.argmax(operation_distribution.probs)
            else:
                operation = torch.multinomial(
                    operation_distribution.probs, 1, generator=self.generator
                ).squeeze(0)
            operation_index = int(operation.item())
            conditional_distribution = _masked_categorical(
                logits, operation_candidate_masks[operation_index]
            )
            if random_uniforms is not None:
                action = self._sample_categorical_with_uniform(
                    conditional_distribution.probs, float(uniforms[1])
                )
            elif deterministic:
                action = torch.argmax(conditional_distribution.probs)
            else:
                action = torch.multinomial(
                    conditional_distribution.probs, 1, generator=self.generator
                ).squeeze(0)
            action_log_prob = operation_distribution.log_prob(operation) + (
                conditional_distribution.log_prob(action)
            )
            distribution = _masked_categorical(logits, mask_t)
            if not torch.allclose(
                action_log_prob,
                distribution.log_prob(action),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise FloatingPointError("Hierarchical and joint Micro log-probabilities differ.")
            inference_seconds = time.perf_counter() - inference_started
            return (
                int(action.item()),
                float(action_log_prob.cpu()),
                distribution.probs.cpu().numpy().astype(np.float32, copy=True),
                float(inference_seconds),
            )

    def _sample_categorical_with_uniform(self, probabilities: Any, uniform: float) -> Any:
        """Inverse-CDF categorical draw for a paired common uniform."""

        value = float(uniform)
        if not math.isfinite(value) or not (0.0 <= value < 1.0):
            raise ValueError("Categorical common uniform must be finite and in [0, 1).")
        probs = np.asarray(
            probabilities.detach().cpu().numpy(), dtype=np.float64
        )
        if (
            probs.ndim != 1
            or len(probs) == 0
            or not np.isfinite(probs).all()
            or np.any(probs < 0.0)
        ):
            raise FloatingPointError("Categorical probabilities are malformed for inverse-CDF sampling.")
        total = math.fsum(float(item) for item in probs)
        if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise FloatingPointError("Categorical probabilities do not sum to one.")
        cdf = np.cumsum(probs / total, dtype=np.float64)
        cdf[-1] = 1.0
        selected = int(np.searchsorted(cdf, value, side="right"))
        if selected >= len(probs):
            positive = np.flatnonzero(probs > 0.0)
            if len(positive) == 0:
                raise FloatingPointError("Categorical distribution has no positive-probability action.")
            selected = int(positive[-1])
        if probs[selected] <= 0.0:
            raise FloatingPointError("Inverse-CDF sampling selected a zero-probability action.")
        return torch.as_tensor(selected, dtype=torch.long, device=probabilities.device)

    def _evaluate_micro_actions(
        self,
        observations: Any,
        candidate_features: Any,
        candidate_masks: Any,
        actions: Any,
        uav0_position_conditions: Any,
    ) -> tuple[Any, Any, Any, Any]:
        observations_t = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        features_t = torch.as_tensor(
            candidate_features, dtype=torch.float32, device=self.device
        )
        masks_t = torch.as_tensor(
            candidate_masks, dtype=torch.bool, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        conditions_t = torch.as_tensor(
            uav0_position_conditions, dtype=torch.float32, device=self.device
        )
        batch_size = int(observations_t.shape[0])
        if (
            tuple(features_t.shape[:2]) != (batch_size, 2)
            or tuple(masks_t.shape) != tuple(features_t.shape[:3])
            or tuple(actions_t.shape) != (batch_size, 2)
            or tuple(conditions_t.shape) != (batch_size, 2)
            or int(features_t.shape[-1]) != self.uav_feature_dim
        ):
            raise ValueError("Micro replay tensors do not match the saved MAPPO action schema.")

        batch_head_log_probs: list[Any] = []
        batch_entropies: list[Any] = []
        for batch_index in range(batch_size):
            head_log_probs: list[Any] = []
            head_entropies: list[Any] = []
            for head in range(2):
                head_features = features_t[batch_index, head]
                head_mask = masks_t[batch_index, head]
                condition = (
                    torch.zeros_like(conditions_t[batch_index])
                    if head == 0
                    else conditions_t[batch_index]
                )
                logits = self.policy._micro_candidate_logits(
                    observations_t[batch_index],
                    head_features,
                    head_mask,
                    condition,
                )
                operation_distribution, operation_candidate_masks = (
                    self._micro_group_distribution(logits, head_features, head_mask)
                )
                action = actions_t[batch_index, head]
                action_index = int(action.item())
                if action_index < 0 or action_index >= len(head_mask) or not bool(
                    head_mask[action_index].item()
                ):
                    raise ValueError("Micro action is masked by its behavior-time legal support.")
                group_ids = torch.as_tensor(
                    head_features[:, -2:], dtype=torch.float32, device=self.device
                ).argmax(dim=-1)
                action_group = group_ids[action_index]
                conditional_distribution = _masked_categorical(
                    logits, operation_candidate_masks[int(action_group.item())]
                )
                head_log_probs.append(
                    operation_distribution.log_prob(action_group)
                    + conditional_distribution.log_prob(action)
                )
                head_entropy = operation_distribution.entropy()
                for group_index, group_candidates in enumerate(operation_candidate_masks):
                    if bool(torch.any(group_candidates).item()):
                        group_conditional = _masked_categorical(logits, group_candidates)
                        head_entropy = head_entropy + (
                            operation_distribution.probs[group_index]
                            * group_conditional.entropy()
                        )
                head_entropies.append(head_entropy)
            batch_head_log_probs.append(torch.stack(head_log_probs))
            batch_entropies.append(torch.stack(head_entropies).sum())
        head_log_probs_t = torch.stack(batch_head_log_probs).to(dtype=torch.float32)
        return (
            head_log_probs_t.to(torch.float64).sum(dim=1).to(dtype=torch.float32),
            self.policy.micro_value(observations_t),
            head_log_probs_t,
            torch.stack(batch_entropies).to(dtype=torch.float32),
        )

    def collect_episode(
        self,
        aligned_steps: list[Any],
        *,
        max_slots: int | None = None,
        terminated_at_end: bool | None = None,
        deterministic: bool = False,
        profile_slot_end_to_end: bool = False,
        episode_seed: int | None = None,
    ) -> MAPPORollout:
        """Collect real slots; a supplied extra aligned step bootstraps truncation."""

        if not aligned_steps:
            raise ValueError("MAPPO rollout requires at least one aligned step.")
        if episode_seed is not None and (
            isinstance(episode_seed, bool)
            or not isinstance(episode_seed, (int, np.integer))
            or int(episode_seed) < 0
        ):
            raise ValueError("episode_seed must be a non-negative integer when provided.")
        if self.stage3_cache_value_config is not None and episode_seed is None:
            raise ValueError("Stage3 counterfactual traces require the source Training episode seed.")
        behavior_policy_sha256 = (
            policy_state_fingerprint(self.policy)
            if self.stage3_cache_value_config is not None
            else None
        )
        transition_slots = (
            int(self.context.episode_slots) if max_slots is None else int(max_slots)
        )
        if transition_slots <= 0 or transition_slots > int(self.context.episode_slots):
            raise ValueError("max_slots must be within the formal Episode length.")
        if len(aligned_steps) < transition_slots:
            raise ValueError("aligned_steps does not include every requested transition slot.")
        if terminated_at_end is None:
            terminated = (
                transition_slots == int(self.context.episode_slots)
                and len(aligned_steps) == transition_slots
            )
        else:
            terminated = bool(terminated_at_end)
        if not terminated and len(aligned_steps) <= transition_slots:
            raise ValueError("A truncated rollout needs one extra aligned bootstrap step.")
        macro_count = (transition_slots + MACRO_INTERVAL_SLOTS - 1) // MACRO_INTERVAL_SLOTS
        selected_counterfactual_macros = set(
            self._selected_counterfactual_macro_indices(macro_count)
        )

        macro_transitions: list[MAPPOMacroTransition] = []
        micro_transitions: list[MAPPOMicroTransition] = []
        counterfactual_traces: list[CounterfactualMacroTrace] = []
        cache = np.asarray(self.context.initial_cache, dtype=np.int8).copy()
        uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in self.context.initial_uav.items()
        }
        active_cache: np.ndarray | None = None
        frame: dict[str, Any] | None = None
        frame_reward = 0.0
        frame_duration = 0
        macro_decision_slots: list[int] = []
        macro_durations: list[int] = []
        cache_updates_per_frame: list[int] = []
        uav_moves_per_slot: list[int] = []
        rewards: list[float] = []
        slot_end_to_end_seconds: list[float] = []
        reward_contract_error_max = 0.0
        kkt_solve_count = 0
        kkt_solver_call_count = 0
        matching_summary_samples: list[dict[str, Any]] = []

        for slot_index in range(transition_slots):
            slot_started = time.perf_counter() if profile_slot_end_to_end else None
            step = aligned_steps[slot_index]
            if slot_index % MACRO_INTERVAL_SLOTS == 0:
                macro_index = len(macro_decision_slots)
                macro_observation = _observation(cache, uav, step, self.context)
                node_candidates, candidate_features, candidate_masks = (
                    _cache_node_candidates(
                        cache, uav, step, self.context, self.tensor_spec
                    )
                )
                cache_sample = self._sample_macro_action(
                    macro_observation,
                    candidate_features,
                    candidate_masks,
                    deterministic=deterministic,
                )
                candidate_masks = np.asarray(cache_sample["masks"], dtype=bool)
                actions = np.asarray(cache_sample["actions"], dtype=np.int64)
                if not np.all(candidate_masks[np.arange(9), actions]):
                    raise ValueError("Macro sampler selected an illegal Cache action.")
                active_cache = cache.copy()
                for node_index, candidate_index in enumerate(actions):
                    active_cache[node_index] = node_candidates[node_index][
                        int(candidate_index)
                    ][node_index]
                cache_configuration.validate_cache_shape_and_capacity(
                    active_cache,
                    self.context.service_catalog,
                    self.context.cache_nodes,
                )
                counterfactual_capture = None
                if (
                    self.stage3_cache_value_config is not None
                    and macro_index in selected_counterfactual_macros
                ):
                    counterfactual_capture = {
                        "macro_index": macro_index,
                        "pre_cache": cache.copy(),
                        "pre_uav": {
                            key: value.copy() for key, value in uav.items()
                        },
                        "candidate_configurations": node_candidates,
                        "aligned_steps": [],
                        "micro_common_uniforms": [],
                    }
                frame = {
                    "slot": int(step.slot),
                    "observation": macro_observation.copy(),
                    "candidate_features": np.asarray(
                        candidate_features, dtype=np.float32
                    ).copy(),
                    "candidate_masks": candidate_masks.copy(),
                    "actions": actions.copy(),
                    "executed_cache_configuration": active_cache.copy(),
                    "conditional_candidate_probabilities": (
                        None
                        if counterfactual_capture is None
                        or cache_sample["conditional_candidate_probabilities"] is None
                        else cache_sample["conditional_candidate_probabilities"].copy()
                    ),
                    "trigger_log_probs": cache_sample["trigger_log_probs"].copy(),
                    "content_log_probs": cache_sample["content_log_probs"].copy(),
                    "node_log_probs": cache_sample["node_log_probs"].copy(),
                    "joint_log_prob": float(cache_sample["joint_log_prob"]),
                    "old_value": float(cache_sample["value"]),
                    "actual_update_by_node": np.zeros(9, dtype=bool),
                    "counterfactual_capture": counterfactual_capture,
                    "actor_inference_seconds": float(
                        cache_sample["actor_inference_seconds"]
                    ),
                }
                frame_reward = 0.0
                frame_duration = 0
                macro_decision_slots.append(int(step.slot))

            if active_cache is None or frame is None:
                raise RuntimeError("Fixed Macro Cache frame was not initialized.")
            capture = frame["counterfactual_capture"]
            if capture is not None:
                capture["aligned_steps"].append(step)
                if self.counterfactual_micro_rng is None:
                    raise RuntimeError(
                        "Stage3 trace capture has no independent Micro common-random stream."
                    )
                capture["micro_common_uniforms"].append(
                    np.asarray(
                        [
                            [self.counterfactual_micro_rng.random() for _ in range(2)]
                            for _ in range(2)
                        ],
                        dtype=np.float64,
                    )
                )
            previous_uav = {key: value.copy() for key, value in uav.items()}
            observation = _observation(active_cache, uav, step, self.context)
            current_trial = {key: value.copy() for key, value in uav.items()}
            previous_positions = {key: value.copy() for key, value in previous_uav.items()}
            feature_lists: list[np.ndarray] = []
            position_lists: list[np.ndarray] = []
            candidate_masks: list[np.ndarray] = []
            actions: list[int] = []
            head_log_probs: list[float] = []
            head_probabilities: list[np.ndarray] = []
            head_inference_seconds: list[float] = []
            valid_candidate_counts: list[int] = []
            uav0_condition = np.zeros(2, dtype=np.float32)
            triggers = {str(uav_id): 1 for uav_id in self.context.action_spec.uav_ids}

            for uav_index, uav_id_raw in enumerate(self.context.action_spec.uav_ids):
                uav_id = str(uav_id_raw)
                candidates, features = _uav_candidates(
                    uav_index,
                    current_trial,
                    previous_positions,
                    active_cache,
                    step,
                    self.context,
                )
                mask = np.ones(len(candidates), dtype=bool)
                condition = (
                    np.zeros(2, dtype=np.float32)
                    if uav_index == 0
                    else uav0_condition.copy()
                )
                action, log_prob, probabilities, inference_seconds = self._sample_micro_head(
                    observation,
                    features,
                    mask,
                    uav_index=uav_index,
                    uav0_position_condition=condition,
                    deterministic=deterministic,
                )
                if action < 0 or action >= len(candidates) or not bool(mask[action]):
                    raise ValueError("Micro sampler selected an illegal UAV candidate.")
                chosen = candidates[action]
                uav_configuration.apply_uav_candidate(
                    current_trial,
                    previous_positions,
                    uav_id,
                    chosen.target_position,
                    triggers,
                    self.context.cfg,
                    self.context.flight_boundary,
                )
                if uav_index == 0:
                    uav0_condition = _positions_condition(
                        current_trial[uav_id], self.context
                    )
                actions.append(action)
                head_log_probs.append(log_prob)
                head_probabilities.append(probabilities)
                head_inference_seconds.append(inference_seconds)
                feature_lists.append(np.asarray(features, dtype=np.float32))
                position_lists.append(
                    np.asarray(
                        [candidate.target_position for candidate in candidates],
                        dtype=np.float32,
                    )
                )
                candidate_masks.append(mask.copy())
                valid_candidate_counts.append(len(candidates))

            uav_configuration.validate_uav_configuration_transition(
                previous_positions,
                current_trial,
                {
                    uav_id: int(
                        np.linalg.norm(current_trial[uav_id] - previous_positions[uav_id])
                        > float(
                            uav_configuration.uav_search_config_from_project(
                                self.context.cfg
                            ).position_tolerance_m
                        )
                    )
                    for uav_id in self.context.action_spec.uav_ids
                },
                self.context.cfg,
                self.context.flight_boundary,
            )
            padded_features, padded_masks, padded_positions = _pad_micro_candidates(
                feature_lists,
                position_lists,
                self.max_uav_candidates,
                self.uav_feature_dim,
            )
            for head, valid_count in enumerate(valid_candidate_counts):
                if not np.array_equal(
                    padded_masks[head, :valid_count], candidate_masks[head]
                ):
                    raise ValueError("Saved Micro action mask differs from the sampling mask.")
                padded_probabilities = np.zeros(self.max_uav_candidates, dtype=np.float32)
                padded_probabilities[:valid_count] = head_probabilities[head]
                if (
                    not np.isfinite(padded_probabilities).all()
                    or np.any(padded_probabilities[~padded_masks[head]] != 0.0)
                    or not np.isclose(padded_probabilities.sum(), 1.0, atol=1.0e-6)
                ):
                    raise FloatingPointError("Micro probabilities do not match saved legal masks.")

            cache_before = cache.copy()
            performance_profile: dict[str, Any] = {}
            result = slot_environment.execute_configured_slot(
                cache_before,
                previous_uav,
                active_cache,
                current_trial,
                step.scenario_slice,
                step.actual_tasks,
                step.actual_vehicle_positions,
                self.context.service_catalog,
                self.context.cfg,
                self.context.cache_nodes,
                self.context.cache_cfg,
                self.context.flight_boundary,
                slot_id=int(step.slot),
                time_s=float(step.time_s),
                performance_profile=performance_profile,
            )
            if not result.step_feasible or result.system_utility is None:
                raise RuntimeError(
                    "Formal Matching+KKT rejected a MAPPO action: "
                    f"slot={step.slot}, reason={result.infeasible_reason}"
                )
            _check_request_contract(cache_before, previous_uav, result, self.context.action_spec)
            reward = _check_reward(result)
            next_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
            next_uav = {
                key: np.asarray(value, dtype=np.float64).copy()
                for key, value in result.next_uav_positions.items()
            }
            if not np.array_equal(next_cache, active_cache):
                raise ValueError("Formal environment changed the selected Cache target.")
            executed_target_positions = np.stack(
                [
                    np.asarray(result.next_uav_positions[uav_id], dtype=np.float32)
                    for uav_id in self.context.action_spec.uav_ids
                ]
            )
            if not np.allclose(
                executed_target_positions,
                np.stack(
                    [
                        padded_positions[head, actions[head]]
                        for head in range(2)
                    ]
                ),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError("Saved Micro action index differs from the executed UAV target.")
            terminal_slot = terminated and slot_index == transition_slots - 1
            if terminal_slot:
                next_observation = np.zeros_like(observation)
            else:
                next_observation = _observation(
                    next_cache, next_uav, aligned_steps[slot_index + 1], self.context
                )

            actual_cache_update = np.any(cache_before != active_cache, axis=1)
            frame["actual_update_by_node"] |= actual_cache_update
            movement_tolerance = float(
                uav_configuration.uav_search_config_from_project(
                    self.context.cfg
                ).position_tolerance_m
            )
            actual_uav_moves = np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(result.next_uav_positions[uav_id], dtype=np.float64)
                        - previous_uav[uav_id]
                    )
                    > movement_tolerance
                    for uav_id in self.context.action_spec.uav_ids
                ],
                dtype=bool,
            )
            if not np.array_equal(actual_uav_moves, np.asarray(actions) != 0):
                raise ValueError("Micro STAY/MOVE actions differ from executed UAV movement.")
            with torch.no_grad():
                micro_value = float(self.policy.micro_value(observation).cpu())
            joint_micro_log_prob = float(math.fsum(head_log_probs))
            micro_transitions.append(
                MAPPOMicroTransition(
                    slot=int(step.slot),
                    observation=observation.copy(),
                    candidate_features=padded_features.copy(),
                    candidate_masks=padded_masks.copy(),
                    candidate_positions=padded_positions.copy(),
                    actions=np.asarray(actions, dtype=np.int64),
                    uav0_position_condition=uav0_condition.copy(),
                    executed_target_positions=executed_target_positions.copy(),
                    old_head_log_probs=np.asarray(head_log_probs, dtype=np.float32),
                    old_joint_log_prob=joint_micro_log_prob,
                    old_value=micro_value,
                    reward=reward,
                    next_observation=next_observation.copy(),
                    done=terminal_slot,
                    actual_uav_moves=actual_uav_moves,
                    matching_summary=dict(result.matching_summary),
                    actual_task_utility=float(result.actual_task_utility),
                    cache_cost=float(result.cache_cost),
                    movement_cost=float(result.movement_cost),
                    movement_energy_j=float(result.movement_energy),
                    matching_kkt_seconds=float(
                        performance_profile.get("matching_kkt_seconds", 0.0)
                    ),
                    actor_inference_seconds=float(sum(head_inference_seconds)),
                )
            )
            frame_reward += (self.gamma ** frame_duration) * reward
            frame_duration += 1
            rewards.append(reward)
            reward_contract_error_max = max(
                reward_contract_error_max,
                abs(float(result.system_utility) - reward),
            )
            kkt_solve_count += int(result.matching_summary.get("KKT_SOLVE_COUNT", 0))
            kkt_solver_call_count += int(
                result.matching_summary.get("KKT_SOLVER_CALL_COUNT", 0)
            )
            matching_summary_samples.append(dict(result.matching_summary))
            cache_updates_per_frame.append(int(np.count_nonzero(actual_cache_update)))
            uav_moves_per_slot.append(int(np.count_nonzero(actual_uav_moves)))
            cache = next_cache
            uav = next_uav

            frame_finished = (
                frame_duration == MACRO_INTERVAL_SLOTS
                or slot_index == transition_slots - 1
            )
            if frame_finished:
                frame_done = terminal_slot
                if not np.array_equal(
                    frame["actual_update_by_node"], frame["actions"] > 0
                ):
                    raise ValueError("Macro KEEP/UPDATE actions differ from executed Cache changes.")
                if frame_done:
                    macro_next_observation = np.zeros_like(frame["observation"])
                    macro_next_value = 0.0
                else:
                    macro_next_observation = next_observation.copy()
                    with torch.no_grad():
                        macro_next_value = float(
                            self.policy.macro_value(macro_next_observation).cpu()
                        )
                macro_transitions.append(
                    MAPPOMacroTransition(
                        slot=int(frame["slot"]),
                        observation=frame["observation"].copy(),
                        candidate_features=frame["candidate_features"].copy(),
                        candidate_masks=frame["candidate_masks"].copy(),
                        actions=frame["actions"].copy(),
                        old_trigger_log_probs=frame["trigger_log_probs"].copy(),
                        old_content_log_probs=frame["content_log_probs"].copy(),
                        old_node_log_probs=frame["node_log_probs"].copy(),
                        old_joint_log_prob=float(frame["joint_log_prob"]),
                        old_value=float(frame["old_value"]),
                        executed_cache_configuration=np.asarray(
                            frame["executed_cache_configuration"], dtype=np.int8
                        ).copy(),
                        frame_reward=float(frame_reward),
                        next_observation=macro_next_observation.copy(),
                        old_next_value=float(macro_next_value),
                        duration=int(frame_duration),
                        done=frame_done,
                        actual_cache_update_by_node=np.asarray(
                            frame["actual_update_by_node"], dtype=bool
                        ).copy(),
                        actor_inference_seconds=float(frame["actor_inference_seconds"]),
                    )
                )
                if capture is not None:
                    conditional_candidate_probabilities = frame[
                        "conditional_candidate_probabilities"
                    ]
                    if conditional_candidate_probabilities is None:
                        raise RuntimeError(
                            "Stage3 counterfactual frame is missing frozen conditional Candidate probabilities."
                        )
                    trace_steps = list(capture["aligned_steps"])
                    frame_truncated = (
                        not frame_done and slot_index == transition_slots - 1
                    )
                    if not frame_done:
                        bootstrap_index = slot_index + 1
                        if bootstrap_index >= len(aligned_steps):
                            raise ValueError(
                                "Selected counterfactual Macro frame has no exact bootstrap Slot."
                            )
                        trace_steps.append(aligned_steps[bootstrap_index])
                    counterfactual_traces.append(
                        make_counterfactual_trace(
                            episode_key=(
                                f"train_seed_{episode_seed}_"
                                if episode_seed is not None
                                else "train_seed_unknown_"
                            )
                            + (
                                f"origins_{aligned_steps[0].origin_index}_"
                                f"through_{aligned_steps[transition_slots - 1].origin_index}"
                            ),
                            episode_seed=(
                                None if episode_seed is None else int(episode_seed)
                            ),
                            behavior_policy_sha256=str(behavior_policy_sha256),
                            rollout_index=self.counterfactual_rollout_count,
                            macro_index=int(capture["macro_index"]),
                            slot=int(frame["slot"]),
                            duration=int(frame_duration),
                            done=bool(frame_done),
                            truncated=bool(frame_truncated),
                            pre_cache=capture["pre_cache"],
                            pre_uav=capture["pre_uav"],
                            executed_cache=frame["executed_cache_configuration"],
                            macro_actions=frame["actions"],
                            behavior_trigger_log_probs=frame["trigger_log_probs"],
                            behavior_content_log_probs=frame["content_log_probs"],
                            behavior_trigger_probabilities=np.exp(
                                frame["trigger_log_probs"]
                            ),
                            behavior_content_probabilities=np.exp(
                                frame["content_log_probs"]
                            ),
                            behavior_conditional_candidate_probabilities=(
                                conditional_candidate_probabilities
                            ),
                            candidate_configurations=capture["candidate_configurations"],
                            candidate_features=frame["candidate_features"],
                            candidate_masks=frame["candidate_masks"],
                            target_node_indices=self._counterfactual_target_nodes(
                                int(capture["macro_index"])
                            ),
                            aligned_steps=trace_steps,
                            micro_common_uniforms=capture["micro_common_uniforms"],
                        )
                    )
                macro_durations.append(frame_duration)
                frame = None
                active_cache = None
                frame_reward = 0.0
                frame_duration = 0
            if slot_started is not None:
                slot_elapsed = time.perf_counter() - slot_started
                if not math.isfinite(slot_elapsed) or slot_elapsed <= 0.0:
                    raise FloatingPointError(
                        f"Invalid end-to-end Slot runtime at slot={step.slot}: {slot_elapsed!r}"
                    )
                slot_end_to_end_seconds.append(float(slot_elapsed))

        cache_actions = np.stack([item.actions for item in macro_transitions])
        cache_update_candidate_counts: dict[str, dict[str, int]] = {}
        for node_index, node_id in enumerate(self.context.action_spec.cache_node_ids):
            selected = cache_actions[:, node_index]
            update_values, update_counts = np.unique(
                selected[selected > 0], return_counts=True
            )
            cache_update_candidate_counts[str(node_id)] = {
                str(int(value)): int(count)
                for value, count in zip(update_values, update_counts, strict=True)
            }
        micro_actions = np.stack([item.actions for item in micro_transitions])
        joint_uav_action_counts: dict[str, int] = {}
        for action_pair in micro_actions:
            name = "/".join("MOVE" if int(value) != 0 else "STAY" for value in action_pair)
            joint_uav_action_counts[name] = joint_uav_action_counts.get(name, 0) + 1
        summary = {
            "baseline_schema": MAPPO_BASELINE_SCHEMA,
            "macro_interval_slots": MACRO_INTERVAL_SLOTS,
            "transition_slots": int(transition_slots),
            "terminated": bool(terminated),
            "truncated": not bool(terminated),
            "macro_decision_slots": macro_decision_slots,
            "macro_actual_durations": macro_durations,
            "macro_update_node_counts_by_slot": cache_updates_per_frame,
            "micro_move_counts_by_slot": uav_moves_per_slot,
            "cache_actions_legal": all(
                bool(np.all(item.candidate_masks[
                    np.arange(9), item.actions
                ]))
                for item in macro_transitions
            ),
            "uav_actions_legal": all(
                all(bool(item.candidate_masks[index, action]) for index, action in enumerate(item.actions))
                for item in micro_transitions
            ),
            "reward_contract_max_abs_error": float(reward_contract_error_max),
            "matching_kkt_solve_count": int(kkt_solve_count),
            "matching_kkt_solver_call_count": int(kkt_solver_call_count),
            "matching_kkt_seconds_total": float(
                sum(item.matching_kkt_seconds for item in micro_transitions)
            ),
            "matching_kkt_seconds_mean": float(
                np.mean([item.matching_kkt_seconds for item in micro_transitions])
            ),
            "slot_end_to_end_seconds": slot_end_to_end_seconds,
            "actor_inference_macro_seconds_mean": float(
                np.mean([item.actor_inference_seconds for item in macro_transitions])
            ),
            "actor_inference_micro_seconds_mean": float(
                np.mean([item.actor_inference_seconds for item in micro_transitions])
            ),
            "actual_net_utility_total": float(sum(rewards)),
            "actual_net_utility_mean": float(np.mean(rewards)),
            "cache_update_cost_total": float(sum(item.cache_cost for item in micro_transitions)),
            "uav_movement_cost_total": float(
                sum(item.movement_cost for item in micro_transitions)
            ),
            "cache_keep_count": int(np.count_nonzero(cache_actions == 0)),
            "cache_update_count": int(np.count_nonzero(cache_actions > 0)),
            "cache_keep_rate": float(np.mean(cache_actions == 0)),
            "cache_update_rate": float(np.mean(cache_actions > 0)),
            "cache_update_candidate_counts_by_node": cache_update_candidate_counts,
            "uav_stay_count": int(np.count_nonzero(micro_actions == 0)),
            "uav_move_count": int(np.count_nonzero(micro_actions != 0)),
            "uav_stay_rate": float(np.mean(micro_actions == 0)),
            "uav_move_rate": float(np.mean(micro_actions != 0)),
            "uav_move_feasible": bool(
                all(np.array_equal(item.actual_uav_moves, item.actions != 0) for item in micro_transitions)
            ),
            "uav_joint_action_counts": joint_uav_action_counts,
            "matching_summary_samples": matching_summary_samples,
            "slot_rewards": rewards,
            "training_window_is_complete_episode": bool(terminated and transition_slots == int(self.context.episode_slots)),
            "counterfactual_trace_count": len(counterfactual_traces),
            "counterfactual_target_nodes": [
                list(trace.target_node_indices) for trace in counterfactual_traces
            ],
            "counterfactual_trace_manifest": [
                {
                    "episode_key": trace.episode_key,
                    "episode_seed": trace.episode_seed,
                    "macro_index": int(trace.macro_index),
                    "slot": int(trace.slot),
                    "duration": int(trace.duration),
                    "done": bool(trace.done),
                    "truncated": bool(trace.truncated),
                    "bootstrap_slot": (
                        None
                        if trace.done
                        else int(trace.aligned_steps[trace.duration].slot)
                    ),
                    "target_node_indices": list(trace.target_node_indices),
                    "behavior_action_indices": trace.macro_actions.tolist(),
                    "behavior_trigger_probabilities": (
                        trace.behavior_trigger_probabilities.tolist()
                    ),
                    "behavior_content_probabilities": (
                        trace.behavior_content_probabilities.tolist()
                    ),
                    "legal_candidate_counts_by_node": [
                        int(np.count_nonzero(mask)) for mask in trace.candidate_masks
                    ],
                    "snapshot_sha256": trace.snapshot_sha256,
                }
                for trace in counterfactual_traces
            ],
        }
        if not summary["cache_actions_legal"] or not summary["uav_actions_legal"]:
            raise ValueError("Collected rollout contains an illegal Cache/UAV action.")
        if reward_contract_error_max > 1.0e-12:
            raise ValueError("Collected reward differs from formal net system utility.")
        if sum(macro_durations) != transition_slots or any(
            duration not in (1, MACRO_INTERVAL_SLOTS) for duration in macro_durations
        ):
            raise ValueError("Macro frame durations do not cover the collected slots.")
        if any(
            duration != MACRO_INTERVAL_SLOTS
            for duration in macro_durations[:-1]
        ):
            raise ValueError("Only the final truncated/terminal Macro frame may be partial.")
        if self.stage3_cache_value_config is not None:
            if policy_state_fingerprint(self.policy) != behavior_policy_sha256:
                raise RuntimeError("MAPPO policy changed during counterfactual trace collection.")
            if len(counterfactual_traces) != len(selected_counterfactual_macros):
                raise ValueError("Stage3 did not capture every selected Macro replay trace.")
            self.counterfactual_rollout_count += 1
        return MAPPORollout(
            macro=tuple(macro_transitions),
            micro=tuple(micro_transitions),
            summary=summary,
            counterfactual_traces=tuple(counterfactual_traces),
        )

    def _build_counterfactual_training_batch(
        self, collected: MAPPORollout
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Generate bounded, exact branch labels before any PPO parameter update."""

        if self.stage3_cache_value_config is None:
            raise RuntimeError("Stage3 counterfactual mode is not enabled.")
        if not collected.counterfactual_traces:
            raise CounterfactualLabelBatchError(
                "Stage3 update requires selected counterfactual Macro traces.",
                {
                    "valid_trace_count": 0,
                    "valid_label_count": 0,
                    "invalid_trace_rejection_count": 0,
                    "invalid_label_rejection_count": 0,
                    "replay_seconds": 0.0,
                    "legal_update_candidate_count": 0,
                    "trigger_supervision_row_count": 0,
                    "trigger_tie_excluded_row_count": 0,
                    "trigger_delta_q_positive_count": 0,
                    "trigger_delta_q_negative_count": 0,
                    "trigger_delta_q_near_zero_count": 0,
                    "candidate_delta_q_positive_count": 0,
                    "candidate_delta_q_negative_count": 0,
                    "candidate_delta_q_near_zero_count": 0,
                    "candidate_pair_total_count": 0,
                    "candidate_pair_tie_count": 0,
                    "candidate_pair_valid_count": 0,
                    "delta_q_huber_scalar_count": 0,
                    "label_diagnostics": [],
                    "rejection_reason_counts": {},
                },
            )
        transitions_by_slot = {int(item.slot): item for item in collected.macro}
        records: list[dict[str, Any]] = []
        branch_execution_count = 0
        branch_cache_reuse_count = 0
        replay_seconds = 0.0
        valid_trace_count = 0
        invalid_trace_rejection_count = 0
        invalid_label_rejection_count = 0
        rejection_reason_counts: dict[str, int] = {}

        def reject(reason: Exception | str, *, labels: int = 1) -> None:
            nonlocal invalid_label_rejection_count
            invalid_label_rejection_count += int(labels)
            key = (
                f"{type(reason).__name__}: {reason}"
                if isinstance(reason, Exception)
                else str(reason)
            )
            rejection_reason_counts[key] = rejection_reason_counts.get(key, 0) + int(labels)

        for trace in collected.counterfactual_traces:
            trace_started = time.perf_counter()
            try:
                transition = transitions_by_slot.get(int(trace.slot))
                if transition is None:
                    raise ValueError(
                        "Counterfactual trace has no matching behavior Macro transition."
                    )
                if (
                    not np.array_equal(transition.actions, trace.macro_actions)
                    or not np.array_equal(
                        transition.executed_cache_configuration, trace.executed_cache
                    )
                    or not np.array_equal(
                        transition.candidate_masks, trace.candidate_masks
                    )
                    or not np.array_equal(
                        transition.candidate_features, trace.candidate_features
                    )
                ):
                    raise ValueError(
                        "Counterfactual trace differs from its behavior Macro transition."
                    )
                result = evaluate_counterfactual_trace(self, trace, gamma=self.gamma)
                if tuple(
                    int(label.node_index) for label in result.labels
                ) != tuple(trace.target_node_indices):
                    raise ValueError(
                        "Counterfactual result target nodes differ from the trace contract."
                    )
                node_features = cache_node_value_features(
                    self.context, trace.pre_cache
                )
            except Exception as exc:
                replay_seconds += time.perf_counter() - trace_started
                invalid_trace_rejection_count += 1
                reject(exc, labels=max(1, len(trace.target_node_indices)))
                continue
            replay_seconds += time.perf_counter() - trace_started
            valid_trace_count += 1
            branch_execution_count += result.branch_execution_count
            branch_cache_reuse_count += result.branch_cache_reuse_count
            trace_records: list[dict[str, Any]] = []
            for label in result.labels:
                node_index = int(label.node_index)
                try:
                    update_mask = np.asarray(
                        trace.candidate_masks[node_index], dtype=bool
                    ).copy()
                    update_mask[0] = False
                    candidate_count = len(update_mask)
                    legal_update_indices = tuple(
                        int(value) for value in np.flatnonzero(update_mask)
                    )
                    if not legal_update_indices:
                        raise ValueError(
                            "Target Cache node has no legal UPDATE candidate."
                        )
                    if tuple(label.candidate_indices) != legal_update_indices:
                        raise ValueError(
                            "Counterfactual label set differs from the legal UPDATE support."
                        )
                    if len(label.candidate_indices) != len(label.candidate_delta_q):
                        raise ValueError(
                            "Counterfactual candidate indices/labels are misaligned."
                        )
                    if label.best_delta_q is None or not math.isfinite(
                        float(label.best_delta_q)
                    ):
                        raise ValueError(
                            "Counterfactual label has no finite legal UPDATE value."
                        )
                    delta_values = np.asarray(label.candidate_delta_q, dtype=np.float64)
                    if not np.isfinite(delta_values).all():
                        raise FloatingPointError(
                            "Counterfactual ΔQ label contains NaN/Inf."
                        )
                    if not math.isclose(
                        float(label.best_delta_q),
                        float(np.max(delta_values)),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    ):
                        raise ValueError(
                            "Counterfactual best ΔQ does not match its legal UPDATE labels."
                        )
                    delta_q_trigger_value = conditional_policy_expected_delta_q(
                        candidate_indices=legal_update_indices,
                        candidate_delta_q=delta_values,
                        behavior_conditional_candidate_probabilities=(
                            trace.behavior_conditional_candidate_probabilities[
                                node_index
                            ]
                        ),
                        legal_candidate_mask=trace.candidate_masks[node_index],
                    )
                    trigger_tie_tolerance = float(
                        self.stage3_cache_value_config.trigger_tie_tolerance
                    )
                    candidate_tie_tolerance = float(
                        self.stage3_cache_value_config.candidate_tie_tolerance
                    )
                    # Match the float32 tensors used by the actual losses so
                    # logged tie and sign counts describe the optimized batch.
                    delta_values_for_loss = np.asarray(delta_values, dtype=np.float32)
                    best_delta_q_for_loss = float(np.float32(label.best_delta_q))
                    best_delta_q_class = (
                        "positive"
                        if best_delta_q_for_loss > trigger_tie_tolerance
                        else "negative"
                        if best_delta_q_for_loss < -trigger_tie_tolerance
                        else "near_zero"
                    )
                    delta_q_trigger_for_loss = float(
                        np.float32(delta_q_trigger_value)
                    )
                    if not math.isfinite(delta_q_trigger_for_loss):
                        raise FloatingPointError(
                            "Frozen Candidate-policy expected Trigger ΔQ is non-finite."
                        )
                    delta_q_trigger_class = (
                        "positive"
                        if delta_q_trigger_for_loss > trigger_tie_tolerance
                        else "negative"
                        if delta_q_trigger_for_loss < -trigger_tie_tolerance
                        else "near_zero"
                    )
                    candidate_delta_q_counts = {
                        "positive": int(
                            np.count_nonzero(
                                delta_values_for_loss > candidate_tie_tolerance
                            )
                        ),
                        "negative": int(
                            np.count_nonzero(
                                delta_values_for_loss < -candidate_tie_tolerance
                            )
                        ),
                        "near_zero": int(
                            np.count_nonzero(
                                np.abs(delta_values_for_loss) <= candidate_tie_tolerance
                            )
                        ),
                    }
                    pair_count = int(len(delta_values_for_loss)) * (
                        int(len(delta_values_for_loss)) - 1
                    ) // 2
                    if pair_count:
                        pair_left, pair_right = np.triu_indices(
                            int(len(delta_values_for_loss)), k=1
                        )
                        pair_differences = (
                            delta_values_for_loss[pair_left]
                            - delta_values_for_loss[pair_right]
                        )
                        pair_tie_count = int(
                            np.count_nonzero(
                                np.abs(pair_differences) <= candidate_tie_tolerance
                            )
                        )
                    else:
                        pair_tie_count = 0
                    node_ids = tuple(
                        str(value) for value in self.context.action_spec.cache_node_ids
                    )
                    node_roles = tuple(
                        str(value).upper()
                        for value in self.context.cache_nodes["node_roles"]
                    )
                    if (
                        len(node_ids) != len(node_roles)
                        or node_index < 0
                        or node_index >= len(node_ids)
                    ):
                        raise ValueError(
                            "Counterfactual target node index does not match the current node contract."
                        )
                    label_diagnostic = {
                        "target_node_id": node_ids[node_index],
                        "target_node_role": node_roles[node_index],
                        "target_node_index": int(node_index),
                        "best_delta_q": best_delta_q_for_loss,
                        "best_delta_q_class": best_delta_q_class,
                        "delta_q_trigger": delta_q_trigger_for_loss,
                        "delta_q_trigger_class": delta_q_trigger_class,
                        "trigger_tie_tolerance": trigger_tie_tolerance,
                        "legal_update_candidate_count": int(len(delta_values_for_loss)),
                        "candidate_delta_q_positive_count": candidate_delta_q_counts[
                            "positive"
                        ],
                        "candidate_delta_q_negative_count": candidate_delta_q_counts[
                            "negative"
                        ],
                        "candidate_delta_q_near_zero_count": candidate_delta_q_counts[
                            "near_zero"
                        ],
                        "candidate_delta_q_near_zero_tolerance": candidate_tie_tolerance,
                        "candidate_pair_total_count": pair_count,
                        "candidate_pair_tie_count": pair_tie_count,
                        "candidate_pair_valid_count": pair_count - pair_tie_count,
                    }
                    delta_targets = np.zeros(candidate_count, dtype=np.float32)
                    for candidate_index, delta_q in zip(
                        label.candidate_indices,
                        delta_values,
                        strict=True,
                    ):
                        if (
                            candidate_index <= 0
                            or candidate_index >= candidate_count
                            or not bool(update_mask[candidate_index])
                        ):
                            raise ValueError(
                                "Counterfactual label is not a legal UPDATE candidate."
                            )
                        delta_targets[candidate_index] = float(delta_q)
                    record = {
                        "observation": np.asarray(
                            transition.observation, dtype=np.float32
                        ),
                        "node_index": node_index,
                        "node_features": np.asarray(
                            node_features[node_index], dtype=np.float32
                        ),
                        "candidate_features": np.asarray(
                            trace.candidate_features[node_index], dtype=np.float32
                        ),
                        "candidate_mask": np.asarray(
                            trace.candidate_masks[node_index], dtype=bool
                        ),
                        "update_mask": update_mask,
                        "delta_targets": delta_targets,
                        "best_delta_q": best_delta_q_for_loss,
                        "trigger_delta_q": delta_q_trigger_for_loss,
                        "has_update": True,
                        "label_diagnostic": label_diagnostic,
                    }
                    if not all(
                        np.isfinite(value).all()
                        for key, value in record.items()
                        if isinstance(value, np.ndarray) and key != "candidate_mask"
                    ):
                        raise FloatingPointError(
                            "Stage3 training record contains NaN/Inf."
                        )
                    trace_records.append(record)
                except Exception as exc:
                    reject(exc)
            records.extend(trace_records)
        valid_label_count = len(records)
        legal_update_candidate_count = sum(
            int(np.count_nonzero(item["update_mask"])) for item in records
        )
        trigger_tie_tolerance = float(
            self.stage3_cache_value_config.trigger_tie_tolerance
        )
        trigger_supervision_row_count = sum(
            int(abs(float(item["trigger_delta_q"])) > trigger_tie_tolerance)
            for item in records
        )
        trigger_tie_excluded_row_count = len(records) - trigger_supervision_row_count
        trigger_delta_q_positive_count = sum(
            int(float(item["trigger_delta_q"]) > trigger_tie_tolerance)
            for item in records
        )
        trigger_delta_q_negative_count = sum(
            int(float(item["trigger_delta_q"]) < -trigger_tie_tolerance)
            for item in records
        )
        trigger_delta_q_near_zero_count = trigger_tie_excluded_row_count
        candidate_pair_total_count = sum(
            int(item["label_diagnostic"]["candidate_pair_total_count"])
            for item in records
        )
        candidate_pair_tie_count = sum(
            int(item["label_diagnostic"]["candidate_pair_tie_count"])
            for item in records
        )
        candidate_pair_valid_count = sum(
            int(item["label_diagnostic"]["candidate_pair_valid_count"])
            for item in records
        )
        candidate_delta_q_positive_count = sum(
            int(item["label_diagnostic"]["candidate_delta_q_positive_count"])
            for item in records
        )
        candidate_delta_q_negative_count = sum(
            int(item["label_diagnostic"]["candidate_delta_q_negative_count"])
            for item in records
        )
        candidate_delta_q_near_zero_count = sum(
            int(item["label_diagnostic"]["candidate_delta_q_near_zero_count"])
            for item in records
        )
        delta_q_huber_scalar_count = sum(
            int(np.count_nonzero(item["update_mask"])) for item in records
        )
        summary = {
            "valid_trace_count": int(valid_trace_count),
            "valid_label_count": int(valid_label_count),
            "invalid_trace_rejection_count": int(invalid_trace_rejection_count),
            "invalid_label_rejection_count": int(invalid_label_rejection_count),
            "branch_execution_count": int(branch_execution_count),
            "branch_cache_reuse_count": int(branch_cache_reuse_count),
            "replay_seconds": float(replay_seconds),
            "legal_update_candidate_count": int(legal_update_candidate_count),
            "trigger_supervision_row_count": int(trigger_supervision_row_count),
            "trigger_tie_excluded_row_count": int(trigger_tie_excluded_row_count),
            "trigger_delta_q_positive_count": int(trigger_delta_q_positive_count),
            "trigger_delta_q_negative_count": int(trigger_delta_q_negative_count),
            "trigger_delta_q_near_zero_count": int(trigger_delta_q_near_zero_count),
            "candidate_delta_q_positive_count": int(
                candidate_delta_q_positive_count
            ),
            "candidate_delta_q_negative_count": int(
                candidate_delta_q_negative_count
            ),
            "candidate_delta_q_near_zero_count": int(
                candidate_delta_q_near_zero_count
            ),
            "candidate_delta_q_near_zero_tolerance": float(
                self.stage3_cache_value_config.candidate_tie_tolerance
            ),
            "candidate_pair_total_count": int(candidate_pair_total_count),
            "candidate_pair_tie_count": int(candidate_pair_tie_count),
            "candidate_pair_valid_count": int(candidate_pair_valid_count),
            "delta_q_huber_scalar_count": int(delta_q_huber_scalar_count),
            "label_diagnostics": [
                dict(item["label_diagnostic"]) for item in records
            ],
            "rejection_reason_counts": rejection_reason_counts,
            "label_semantics": (
                "finite_horizon_macro_smdp_return_plus_frozen_critic_bootstrap"
            ),
        }
        if not records or legal_update_candidate_count <= 0:
            raise CounterfactualLabelBatchError(
                "Stage3 rejected the batch because it contains no valid legal UPDATE labels.",
                summary,
            )
        max_candidates = max(len(item["candidate_mask"]) for item in records)
        candidate_feature_dim = int(records[0]["candidate_features"].shape[-1])
        batch_size = len(records)
        candidate_features = np.zeros(
            (batch_size, max_candidates, candidate_feature_dim), dtype=np.float32
        )
        candidate_masks = np.zeros((batch_size, max_candidates), dtype=bool)
        update_masks = np.zeros((batch_size, max_candidates), dtype=bool)
        delta_targets = np.zeros((batch_size, max_candidates), dtype=np.float32)
        for index, item in enumerate(records):
            count = len(item["candidate_mask"])
            if item["candidate_features"].shape != (count, candidate_feature_dim):
                raise ValueError("Stage3 target candidate feature dimensions changed in a batch.")
            candidate_features[index, :count] = item["candidate_features"]
            candidate_masks[index, :count] = item["candidate_mask"]
            update_masks[index, :count] = item["update_mask"]
            delta_targets[index, :count] = item["delta_targets"]
        batch = {
            "observations": np.stack([item["observation"] for item in records]).astype(
                np.float32, copy=False
            ),
            "node_indices": np.asarray(
                [item["node_index"] for item in records], dtype=np.int64
            ),
            "node_features": np.stack([item["node_features"] for item in records]).astype(
                np.float32, copy=False
            ),
            "candidate_features": candidate_features,
            "candidate_masks": candidate_masks,
            "update_masks": update_masks,
            "delta_q_targets": delta_targets,
            "delta_q_trigger": np.asarray(
                [item["trigger_delta_q"] for item in records], dtype=np.float32
            ),
            "trigger_masks": np.asarray(
                [item["has_update"] for item in records], dtype=bool
            ),
        }
        summary["target_record_count"] = int(batch_size)
        summary["legal_update_candidate_count"] = int(update_masks.sum())
        return batch, summary

    @staticmethod
    def _pad_macro_batch(
        transitions: tuple[MAPPOMacroTransition, ...],
    ) -> dict[str, np.ndarray]:
        if not transitions:
            raise ValueError("Macro MAPPO batch is empty.")
        batch_size = len(transitions)
        node_count = int(transitions[0].candidate_masks.shape[0])
        feature_dim = int(transitions[0].candidate_features.shape[-1])
        max_candidates = max(int(item.candidate_masks.shape[-1]) for item in transitions)
        features = np.zeros(
            (batch_size, node_count, max_candidates, feature_dim), dtype=np.float32
        )
        masks = np.zeros((batch_size, node_count, max_candidates), dtype=bool)
        for index, item in enumerate(transitions):
            item_features = np.asarray(item.candidate_features, dtype=np.float32)
            item_masks = np.asarray(item.candidate_masks, dtype=bool)
            if item_features.shape != (*item_masks.shape, feature_dim):
                raise ValueError("Macro candidate features/masks lost alignment.")
            if item_masks.shape[0] != node_count:
                raise ValueError("Macro node count changed inside a rollout.")
            count = int(item_masks.shape[-1])
            features[index, :, :count] = item_features
            masks[index, :, :count] = item_masks
        return {
            "observations": np.stack([item.observation for item in transitions]).astype(np.float32),
            "features": features,
            "masks": masks,
            "actions": np.stack([item.actions for item in transitions]).astype(np.int64),
            "old_log_probs": np.asarray(
                [item.old_joint_log_prob for item in transitions], dtype=np.float32
            ),
            "old_node_log_probs": np.stack(
                [item.old_node_log_probs for item in transitions]
            ).astype(np.float32),
            "old_values": np.asarray([item.old_value for item in transitions], dtype=np.float32),
            "rewards": np.asarray([item.frame_reward for item in transitions], dtype=np.float32),
            "next_values": np.asarray(
                [item.old_next_value for item in transitions], dtype=np.float32
            ),
            "durations": np.asarray([item.duration for item in transitions], dtype=np.int64),
            "dones": np.asarray([item.done for item in transitions], dtype=bool),
        }

    @staticmethod
    def _pad_micro_batch(
        transitions: tuple[MAPPOMicroTransition, ...],
    ) -> dict[str, np.ndarray]:
        if not transitions:
            raise ValueError("Micro MAPPO batch is empty.")
        max_candidates = max(int(item.candidate_masks.shape[-1]) for item in transitions)
        feature_dim = int(transitions[0].candidate_features.shape[-1])
        features = np.zeros(
            (len(transitions), 2, max_candidates, feature_dim), dtype=np.float32
        )
        masks = np.zeros((len(transitions), 2, max_candidates), dtype=bool)
        for index, item in enumerate(transitions):
            count = int(item.candidate_masks.shape[-1])
            if item.candidate_features.shape != (2, count, feature_dim):
                raise ValueError("Micro candidate features/masks lost alignment.")
            features[index, :, :count] = item.candidate_features
            masks[index, :, :count] = item.candidate_masks
        return {
            "observations": np.stack([item.observation for item in transitions]).astype(np.float32),
            "features": features,
            "masks": masks,
            "actions": np.stack([item.actions for item in transitions]).astype(np.int64),
            "conditions": np.stack(
                [item.uav0_position_condition for item in transitions]
            ).astype(np.float32),
            "old_log_probs": np.asarray(
                [item.old_joint_log_prob for item in transitions], dtype=np.float32
            ),
            "old_head_log_probs": np.stack(
                [item.old_head_log_probs for item in transitions]
            ).astype(np.float32),
            "old_values": np.asarray([item.old_value for item in transitions], dtype=np.float32),
            "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
            "next_observations": np.stack(
                [item.next_observation for item in transitions]
            ).astype(np.float32),
            "dones": np.asarray([item.done for item in transitions], dtype=bool),
        }

    def update(self, collected: MAPPORollout, *, epochs: int = 1) -> dict[str, Any]:
        """Perform a full-batch joint PPO actor update and centralized critic fits."""

        if torch is None:
            raise ImportError("Stage-1 MAPPO requires PyTorch.")
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive.")
        counterfactual_batch: dict[str, np.ndarray] | None = None
        counterfactual_summary: dict[str, Any] = {}
        if self.stage3_cache_value_config is not None:
            if (
                self.cache_delta_q_value_head is None
                or self.cache_delta_q_value_optimizer is None
            ):
                raise RuntimeError("Stage3 mode is missing its value head or optimizer.")
            counterfactual_batch, counterfactual_summary = (
                self._build_counterfactual_training_batch(collected)
            )
        macro = self._pad_macro_batch(collected.macro)
        micro = self._pad_micro_batch(collected.micro)
        macro_advantages, macro_returns = compute_macro_smdp_gae(
            macro["rewards"],
            macro["old_values"],
            macro["next_values"],
            macro["durations"],
            macro["dones"],
            self.gamma,
            self.gae_lambda,
        )
        final_micro = collected.micro[-1]
        micro_bootstrap = (
            0.0
            if final_micro.done
            else float(self.policy.micro_value(final_micro.next_observation).detach().cpu())
        )
        micro_advantages, micro_returns = compute_gae(
            micro["rewards"],
            micro["old_values"],
            micro["dones"],
            micro_bootstrap,
            self.gamma,
            self.gae_lambda,
        )
        normalized_macro_advantages, macro_adv_mean, macro_adv_std = _normalize_advantages(
            macro_advantages
        )
        normalized_micro_advantages, micro_adv_mean, micro_adv_std = _normalize_advantages(
            micro_advantages
        )

        macro_observations = torch.as_tensor(
            macro["observations"], dtype=torch.float32, device=self.device
        )
        macro_features = torch.as_tensor(
            macro["features"], dtype=torch.float32, device=self.device
        )
        macro_masks = torch.as_tensor(macro["masks"], dtype=torch.bool, device=self.device)
        macro_actions = torch.as_tensor(macro["actions"], dtype=torch.long, device=self.device)
        macro_old_log_probs = torch.as_tensor(
            macro["old_log_probs"], dtype=torch.float32, device=self.device
        )
        macro_old_node_log_probs = torch.as_tensor(
            macro["old_node_log_probs"], dtype=torch.float32, device=self.device
        )
        macro_adv_t = torch.as_tensor(
            normalized_macro_advantages, dtype=torch.float32, device=self.device
        )
        macro_returns_t = torch.as_tensor(
            macro_returns, dtype=torch.float32, device=self.device
        )
        micro_observations = torch.as_tensor(
            micro["observations"], dtype=torch.float32, device=self.device
        )
        micro_features = torch.as_tensor(
            micro["features"], dtype=torch.float32, device=self.device
        )
        micro_masks = torch.as_tensor(micro["masks"], dtype=torch.bool, device=self.device)
        micro_actions = torch.as_tensor(micro["actions"], dtype=torch.long, device=self.device)
        micro_conditions = torch.as_tensor(
            micro["conditions"], dtype=torch.float32, device=self.device
        )
        micro_old_log_probs = torch.as_tensor(
            micro["old_log_probs"], dtype=torch.float32, device=self.device
        )
        micro_old_head_log_probs = torch.as_tensor(
            micro["old_head_log_probs"], dtype=torch.float32, device=self.device
        )
        micro_adv_t = torch.as_tensor(
            normalized_micro_advantages, dtype=torch.float32, device=self.device
        )
        micro_returns_t = torch.as_tensor(
            micro_returns, dtype=torch.float32, device=self.device
        )

        stage3_tensors: dict[str, Any] | None = None
        if counterfactual_batch is not None:
            stage3_tensors = {
                "observations": torch.as_tensor(
                    counterfactual_batch["observations"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "node_indices": torch.as_tensor(
                    counterfactual_batch["node_indices"],
                    dtype=torch.long,
                    device=self.device,
                ),
                "node_features": torch.as_tensor(
                    counterfactual_batch["node_features"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "candidate_features": torch.as_tensor(
                    counterfactual_batch["candidate_features"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "candidate_masks": torch.as_tensor(
                    counterfactual_batch["candidate_masks"],
                    dtype=torch.bool,
                    device=self.device,
                ),
                "update_masks": torch.as_tensor(
                    counterfactual_batch["update_masks"],
                    dtype=torch.bool,
                    device=self.device,
                ),
                "delta_q_targets": torch.as_tensor(
                    counterfactual_batch["delta_q_targets"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "delta_q_trigger": torch.as_tensor(
                    counterfactual_batch["delta_q_trigger"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "trigger_masks": torch.as_tensor(
                    counterfactual_batch["trigger_masks"],
                    dtype=torch.bool,
                    device=self.device,
                ),
            }
        stage3_head_parameters = (
            ()
            if self.cache_delta_q_value_head is None
            else tuple(
                parameter.detach().clone()
                for parameter in self.cache_delta_q_value_head.parameters()
            )
        )

        with torch.no_grad():
            macro_before, _, macro_node_before, _ = self._evaluate_macro_actions(
                macro_observations, macro_features, macro_masks, macro_actions
            )
            micro_before, _, micro_head_before, _ = self._evaluate_micro_actions(
                micro_observations,
                micro_features,
                micro_masks,
                micro_actions,
                micro_conditions,
            )
        macro_old_error = float(torch.max(torch.abs(macro_before - macro_old_log_probs)).cpu())
        micro_old_error = float(torch.max(torch.abs(micro_before - micro_old_log_probs)).cpu())
        macro_node_old_error = float(
            torch.max(torch.abs(macro_node_before - macro_old_node_log_probs)).cpu()
        )
        micro_head_old_error = float(
            torch.max(torch.abs(micro_head_before - micro_old_head_log_probs)).cpu()
        )
        if max(macro_old_error, micro_old_error, macro_node_old_error, micro_head_old_error) > 2.0e-5:
            raise ValueError(
                "Behavior and replay joint log-probabilities differ: "
                f"macro={macro_old_error}, macro_node={macro_node_old_error}, "
                f"micro={micro_old_error}, micro_head={micro_head_old_error}."
            )

        actor_parameters = (*self.macro_actor_parameters, *self.micro_actor_parameters)
        critic_parameters = (*self.macro_critic_parameters, *self.micro_critic_parameters)
        before_parameters = {
            "macro_actor": tuple(value.detach().clone() for value in self.macro_actor_parameters),
            "macro_candidate": tuple(value.detach().clone() for value in self.macro_candidate_parameters),
            "micro_actor": tuple(value.detach().clone() for value in self.micro_actor_parameters),
            "macro_critic": tuple(value.detach().clone() for value in self.macro_critic_parameters),
            "micro_critic": tuple(value.detach().clone() for value in self.micro_critic_parameters),
        }
        if self.cache_delta_q_value_head is not None:
            before_parameters["cache_delta_q_value"] = stage3_head_parameters
        macro_actor_losses: list[float] = []
        micro_actor_losses: list[float] = []
        macro_critic_losses: list[float] = []
        micro_critic_losses: list[float] = []
        macro_actor_grad_norms: list[float] = []
        macro_actor_total_losses: list[float] = []
        micro_actor_grad_norms: list[float] = []
        macro_critic_grad_norms: list[float] = []
        micro_critic_grad_norms: list[float] = []
        stage3_trigger_losses: list[float] = []
        stage3_candidate_losses: list[float] = []
        stage3_candidate_grad_norms: list[float] = []
        stage3_candidate_parameter_deltas: list[float] = []
        stage3_value_losses: list[float] = []
        stage3_value_grad_norms: list[float] = []
        stage3_value_scalar_counts: list[int] = []
        stage3_trigger_comparison_counts: list[int] = []
        stage3_candidate_pair_counts: list[int] = []
        stage3_ppo_trigger_gradient_relation: dict[str, Any] = {
            "status": "UNRESOLVED",
            "reason": "Stage3 Trigger gradient diagnostics were not evaluated.",
        }
        for epoch_index in range(int(epochs)):
            self.macro_actor_optimizer.zero_grad(set_to_none=True)
            macro_log_probs, _, _, _ = self._evaluate_macro_actions(
                macro_observations, macro_features, macro_masks, macro_actions
            )
            macro_ratio = torch.exp(macro_log_probs - macro_old_log_probs)
            macro_actor_loss = _clipped_ppo_surrogate_loss(
                macro_ratio, macro_adv_t, self.clip_epsilon
            )
            if not bool(torch.isfinite(macro_actor_loss).item()):
                raise FloatingPointError("Macro MAPPO actor loss is non-finite.")
            macro_actor_total_loss = macro_actor_loss
            stage3_trigger_loss = macro_actor_loss.new_zeros(())
            stage3_candidate_loss = macro_actor_loss.new_zeros(())
            if stage3_tensors is not None:
                stage3_config = self.stage3_cache_value_config
                if stage3_config is None:
                    raise RuntimeError("Stage3 tensors exist without a Stage3 configuration.")
                (
                    trigger_logits,
                    _,
                    conditional_logits,
                    _,
                    _,
                    legal_masks,
                ) = self.policy._macro_cache_factorized_node_logits(
                    stage3_tensors["observations"],
                    stage3_tensors["candidate_features"].unsqueeze(1),
                    stage3_tensors["candidate_masks"].unsqueeze(1),
                    detach_candidate_context=True,
                )
                batch_indices = torch.arange(
                    int(stage3_tensors["node_indices"].shape[0]),
                    dtype=torch.long,
                    device=self.device,
                )
                node_indices = stage3_tensors["node_indices"]
                selected_trigger_logits = trigger_logits[batch_indices, 0]
                selected_candidate_logits = conditional_logits[batch_indices, 0]
                selected_masks = legal_masks[batch_indices, 0]
                if not torch.equal(selected_masks, stage3_tensors["candidate_masks"]):
                    raise ValueError("Stage3 replay legal candidates differ from the Macro actor mask.")
                update_legal = selected_masks[:, 1:]
                has_update = torch.any(update_legal, dim=-1)
                update_group_logit = torch.logsumexp(
                    selected_trigger_logits[:, 1:].masked_fill(
                        ~update_legal, -float("inf")
                    ),
                    dim=-1,
                )
                update_log_odds = torch.where(
                    has_update,
                    update_group_logit - selected_trigger_logits[:, 0],
                    torch.zeros_like(update_group_logit),
                )
                stage3_trigger_loss, trigger_count = trigger_value_ranking_loss(
                    update_log_odds,
                    stage3_tensors["delta_q_trigger"],
                    stage3_tensors["trigger_masks"] & has_update,
                    tie_tolerance=stage3_config.trigger_tie_tolerance,
                )
                stage3_actor_auxiliary = (
                    float(stage3_config.trigger_ranking_coefficient)
                    * stage3_trigger_loss
                )
                macro_actor_total_loss = macro_actor_loss + stage3_actor_auxiliary
                stage3_trigger_comparison_counts.append(int(trigger_count))
                stage3_trigger_losses.append(float(stage3_trigger_loss.detach().cpu()))
                if epoch_index == 0:
                    # Read the two gradients on the same rollout and parameter
                    # snapshot. autograd.grad returns tensors without changing
                    # .grad or performing an optimizer step; the normal joint
                    # backward/step below remains the only actor update.
                    try:
                        actor_parameter_ids = {
                            id(parameter) for parameter in self.macro_actor_parameters
                        }
                        actor_named_parameters = tuple(
                            (name, parameter)
                            for name, parameter in self.policy.named_parameters()
                            if id(parameter) in actor_parameter_ids
                        )
                        shared_trigger_named_parameters = tuple(
                            (name, parameter)
                            for name, parameter in actor_named_parameters
                            if name.startswith(
                                (
                                    "macro_actor_encoder.",
                                    "macro_cache_candidate_encoder.",
                                    "macro_cache_scorer.",
                                )
                            )
                        )
                        actor_parameters_for_grad = tuple(
                            parameter for _, parameter in actor_named_parameters
                        )
                        shared_parameters_for_grad = tuple(
                            parameter
                            for _, parameter in shared_trigger_named_parameters
                        )
                        ppo_actor_grads = torch.autograd.grad(
                            macro_actor_loss,
                            actor_parameters_for_grad,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        trigger_actor_grads = torch.autograd.grad(
                            stage3_trigger_loss,
                            actor_parameters_for_grad,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        actor_positions = {
                            id(parameter): index
                            for index, parameter in enumerate(actor_parameters_for_grad)
                        }
                        shared_positions = tuple(
                            actor_positions[id(parameter)]
                            for parameter in shared_parameters_for_grad
                        )

                        def flatten_gradients(
                            parameters: tuple[Any, ...],
                            gradients: tuple[Any, ...],
                            positions: tuple[int, ...] | None = None,
                        ) -> Any:
                            indices = (
                                tuple(range(len(parameters)))
                                if positions is None
                                else positions
                            )
                            parts = []
                            for index in indices:
                                gradient = gradients[index]
                                parameter = parameters[index]
                                parts.append(
                                    torch.zeros_like(parameter).reshape(-1)
                                    if gradient is None
                                    else gradient.detach().reshape(-1)
                                )
                            if not parts:
                                return torch.empty(0, dtype=torch.float32, device=self.device)
                            return torch.cat(parts)

                        ppo_actor_vector = flatten_gradients(
                            actor_parameters_for_grad, ppo_actor_grads
                        )
                        trigger_actor_vector = flatten_gradients(
                            actor_parameters_for_grad, trigger_actor_grads
                        )
                        ppo_shared_vector = flatten_gradients(
                            actor_parameters_for_grad,
                            ppo_actor_grads,
                            shared_positions,
                        )
                        trigger_shared_vector = flatten_gradients(
                            actor_parameters_for_grad,
                            trigger_actor_grads,
                            shared_positions,
                        )
                        ppo_shared_norm = torch.linalg.vector_norm(ppo_shared_vector)
                        trigger_shared_norm = torch.linalg.vector_norm(
                            trigger_shared_vector
                        )
                        ppo_actor_norm = torch.linalg.vector_norm(ppo_actor_vector)
                        trigger_actor_norm = torch.linalg.vector_norm(
                            trigger_actor_vector
                        )
                        shared_overlap = (ppo_shared_vector != 0) & (
                            trigger_shared_vector != 0
                        )
                        overlap_count = int(shared_overlap.sum().item())
                        conflict_count = int(
                            (
                                shared_overlap
                                & (ppo_shared_vector * trigger_shared_vector < 0)
                            ).sum().item()
                        )
                        finite_gradients = all(
                            bool(torch.isfinite(vector).all().item())
                            for vector in (
                                ppo_actor_vector,
                                trigger_actor_vector,
                                ppo_shared_vector,
                                trigger_shared_vector,
                            )
                        )
                        can_compare = (
                            finite_gradients
                            and int(trigger_count) > 0
                            and float(ppo_shared_norm.detach().cpu()) > 0.0
                            and float(trigger_shared_norm.detach().cpu()) > 0.0
                            and overlap_count > 0
                        )
                        if can_compare:
                            cosine = float(
                                (
                                    torch.dot(ppo_shared_vector, trigger_shared_vector)
                                    / (ppo_shared_norm * trigger_shared_norm)
                                )
                                .detach()
                                .cpu()
                            )
                            if cosine < 0.0:
                                direction_relation = "CONFLICTING"
                                conflict_flag = "YES"
                            elif cosine > 0.0 and conflict_count > 0:
                                direction_relation = "MIXED_NET_ALIGNED"
                                conflict_flag = "NO"
                            elif cosine > 0.0:
                                direction_relation = "ALIGNED"
                                conflict_flag = "NO"
                            else:
                                direction_relation = "ORTHOGONAL"
                                conflict_flag = "NO"
                            diagnostic_status = "MEASURED"
                            unresolved_reason = None
                        else:
                            cosine = None
                            direction_relation = "UNRESOLVED"
                            conflict_flag = "UNRESOLVED"
                            diagnostic_status = "UNRESOLVED"
                            unresolved_reason = (
                                "Non-finite gradients, no effective Trigger rows, zero shared gradient, "
                                "or no overlapping nonzero shared coordinates."
                            )
                        stage3_ppo_trigger_gradient_relation = {
                            "status": diagnostic_status,
                            "reason": unresolved_reason,
                            "direction_relation": direction_relation,
                            "ppo_aux_gradient_conflict": conflict_flag,
                            "diagnostic_epoch_index": int(epoch_index),
                            "same_macro_rollout_batch": True,
                            "ppo_macro_actor_loss": float(macro_actor_loss.detach().cpu()),
                            "trigger_auxiliary_loss_unweighted": float(
                                stage3_trigger_loss.detach().cpu()
                            ),
                            "trigger_auxiliary_coefficient": float(
                                stage3_config.trigger_ranking_coefficient
                            ),
                            "ppo_macro_actor_gradient_l2": float(
                                ppo_actor_norm.detach().cpu()
                            ),
                            "trigger_auxiliary_gradient_l2_unweighted": float(
                                trigger_actor_norm.detach().cpu()
                            ),
                            "trigger_auxiliary_gradient_l2_weighted": float(
                                (
                                    trigger_actor_norm
                                    * float(stage3_config.trigger_ranking_coefficient)
                                )
                                .detach()
                                .cpu()
                            ),
                            "shared_trigger_parameter_tensor_count": int(
                                len(shared_trigger_named_parameters)
                            ),
                            "shared_trigger_ppo_gradient_l2": float(
                                ppo_shared_norm.detach().cpu()
                            ),
                            "shared_trigger_auxiliary_gradient_l2_unweighted": float(
                                trigger_shared_norm.detach().cpu()
                            ),
                            "shared_trigger_auxiliary_gradient_l2_weighted": float(
                                (
                                    trigger_shared_norm
                                    * float(stage3_config.trigger_ranking_coefficient)
                                )
                                .detach()
                                .cpu()
                            ),
                            "shared_trigger_gradient_cosine": cosine,
                            "shared_trigger_nonzero_overlap_coordinate_count": overlap_count,
                            "shared_trigger_conflicting_coordinate_count": conflict_count,
                            "shared_trigger_conflict_ratio": (
                                None
                                if overlap_count == 0
                                else float(conflict_count / overlap_count)
                            ),
                            "optimizer_step_called_by_diagnostic": False,
                        }
                    except Exception as exc:
                        stage3_ppo_trigger_gradient_relation = {
                            "status": "UNRESOLVED",
                            "reason": f"Gradient diagnostic failed: {type(exc).__name__}: {exc}",
                            "direction_relation": "UNRESOLVED",
                            "ppo_aux_gradient_conflict": "UNRESOLVED",
                            "optimizer_step_called_by_diagnostic": False,
                        }
            macro_actor_total_loss.backward()
            macro_actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.macro_actor_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(macro_actor_grad_norm).item()):
                raise FloatingPointError("Macro MAPPO actor gradient is non-finite.")
            self.macro_actor_optimizer.step()
            macro_actor_losses.append(float(macro_actor_loss.detach().cpu()))
            macro_actor_total_losses.append(float(macro_actor_total_loss.detach().cpu()))
            macro_actor_grad_norms.append(float(macro_actor_grad_norm.detach().cpu()))

            if stage3_tensors is not None:
                stage3_config = self.stage3_cache_value_config
                if (
                    stage3_config is None
                    or self.cache_delta_q_value_head is None
                    or self.cache_delta_q_value_optimizer is None
                ):
                    raise RuntimeError("Stage3 value-head optimizer is not initialized.")
                self.cache_delta_q_value_optimizer.zero_grad(set_to_none=True)
                delta_q_predictions = self.cache_delta_q_value_head(
                    stage3_tensors["observations"],
                    stage3_tensors["node_features"],
                    stage3_tensors["candidate_features"],
                )
                stage3_value_loss, stage3_value_count = delta_q_huber_loss(
                    delta_q_predictions,
                    stage3_tensors["delta_q_targets"],
                    stage3_tensors["update_masks"],
                    huber_delta=stage3_config.huber_delta,
                )
                if not bool(torch.isfinite(stage3_value_loss).item()):
                    raise FloatingPointError("Stage3 ΔQ Huber loss is non-finite.")
                if (
                    float(stage3_config.delta_q_huber_coefficient) > 0.0
                    and stage3_value_count > 0
                ):
                    stage3_total_value_loss = (
                        float(stage3_config.delta_q_huber_coefficient)
                        * stage3_value_loss
                    )
                    stage3_total_value_loss.backward()
                    stage3_value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.cache_delta_q_value_head.parameters(),
                        self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    if not bool(torch.isfinite(stage3_value_grad_norm).item()):
                        raise FloatingPointError("Stage3 ΔQ value-head gradient is non-finite.")
                    self.cache_delta_q_value_optimizer.step()
                    stage3_value_grad_norms.append(
                        float(stage3_value_grad_norm.detach().cpu())
                    )
                else:
                    stage3_value_grad_norms.append(0.0)
                stage3_value_losses.append(float(stage3_value_loss.detach().cpu()))
                stage3_value_scalar_counts.append(int(stage3_value_count))

            self.micro_actor_optimizer.zero_grad(set_to_none=True)
            micro_log_probs, _, _, _ = self._evaluate_micro_actions(
                micro_observations,
                micro_features,
                micro_masks,
                micro_actions,
                micro_conditions,
            )
            micro_ratio = torch.exp(micro_log_probs - micro_old_log_probs)
            micro_actor_loss = _clipped_ppo_surrogate_loss(
                micro_ratio, micro_adv_t, self.clip_epsilon
            )
            if not bool(torch.isfinite(micro_actor_loss).item()):
                raise FloatingPointError("Micro MAPPO actor loss is non-finite.")
            micro_actor_loss.backward()
            micro_actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.micro_actor_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(micro_actor_grad_norm).item()):
                raise FloatingPointError("Micro MAPPO actor gradient is non-finite.")
            self.micro_actor_optimizer.step()
            micro_actor_losses.append(float(micro_actor_loss.detach().cpu()))
            micro_actor_grad_norms.append(float(micro_actor_grad_norm.detach().cpu()))

            self.macro_critic_optimizer.zero_grad(set_to_none=True)
            macro_values = self.policy.macro_value(macro_observations)
            macro_critic_loss = 0.5 * torch.mean((macro_values - macro_returns_t) ** 2)
            if not bool(torch.isfinite(macro_critic_loss).item()):
                raise FloatingPointError("Macro centralized critic loss is non-finite.")
            macro_critic_loss.backward()
            macro_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.macro_critic_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(macro_critic_grad_norm).item()):
                raise FloatingPointError("Macro centralized critic gradient is non-finite.")
            self.macro_critic_optimizer.step()
            macro_critic_losses.append(float(macro_critic_loss.detach().cpu()))
            macro_critic_grad_norms.append(float(macro_critic_grad_norm.detach().cpu()))

            self.micro_critic_optimizer.zero_grad(set_to_none=True)
            micro_values = self.policy.micro_value(micro_observations)
            micro_critic_loss = 0.5 * torch.mean((micro_values - micro_returns_t) ** 2)
            if not bool(torch.isfinite(micro_critic_loss).item()):
                raise FloatingPointError("Micro centralized critic loss is non-finite.")
            micro_critic_loss.backward()
            micro_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.micro_critic_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(micro_critic_grad_norm).item()):
                raise FloatingPointError("Micro centralized critic gradient is non-finite.")
            self.micro_critic_optimizer.step()
            micro_critic_losses.append(float(micro_critic_loss.detach().cpu()))
            micro_critic_grad_norms.append(float(micro_critic_grad_norm.detach().cpu()))

            # The ranking update follows the complete PPO actor/critic update.
            # Its detached global context and separate head keep this gradient
            # exclusive to the conditional UPDATE-candidate parameters.
            if stage3_tensors is not None:
                stage3_config = self.stage3_cache_value_config
                if stage3_config is None or not self.macro_candidate_parameters:
                    raise RuntimeError("Stage3 conditional Candidate optimizer path is unavailable.")
                candidate_parameters_before = tuple(
                    parameter.detach().clone()
                    for parameter in self.macro_candidate_parameters
                )
                self.macro_actor_optimizer.zero_grad(set_to_none=True)
                (
                    _,
                    _,
                    candidate_logits,
                    _,
                    _,
                    candidate_legal_masks,
                ) = self.policy._macro_cache_factorized_node_logits(
                    stage3_tensors["observations"],
                    stage3_tensors["candidate_features"].unsqueeze(1),
                    stage3_tensors["candidate_masks"].unsqueeze(1),
                    detach_candidate_context=True,
                )
                batch_indices = torch.arange(
                    int(stage3_tensors["node_indices"].shape[0]),
                    dtype=torch.long,
                    device=self.device,
                )
                candidate_scores = candidate_logits[batch_indices, 0]
                candidate_masks_for_loss = candidate_legal_masks[batch_indices, 0]
                if not torch.equal(
                    candidate_masks_for_loss, stage3_tensors["candidate_masks"]
                ):
                    raise ValueError("Stage3 Candidate ranking mask changed after PPO update.")
                stage3_candidate_loss, candidate_pair_count = (
                    candidate_value_pairwise_ranking_loss(
                        candidate_scores.unsqueeze(1),
                        stage3_tensors["delta_q_targets"].unsqueeze(1),
                        stage3_tensors["update_masks"].unsqueeze(1),
                        tie_tolerance=stage3_config.candidate_tie_tolerance,
                        temperature=stage3_config.pairwise_temperature,
                    )
                )
                if not bool(torch.isfinite(stage3_candidate_loss).item()):
                    raise FloatingPointError("Stage3 Candidate ranking loss is non-finite.")
                if (
                    float(stage3_config.candidate_ranking_coefficient) <= 0.0
                    or candidate_pair_count <= 0
                ):
                    raise CounterfactualLabelBatchError(
                        "Stage3 rollout has no effective non-tied legal Candidate pairs.",
                        counterfactual_summary,
                    )
                candidate_total_loss = (
                    float(stage3_config.candidate_ranking_coefficient)
                    * stage3_candidate_loss
                )
                candidate_total_loss.backward()
                candidate_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.macro_candidate_parameters,
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                if not bool(torch.isfinite(candidate_grad_norm).item()):
                    raise FloatingPointError("Stage3 Candidate Actor gradient is non-finite.")
                self.macro_actor_optimizer.step()
                self.candidate_supervised_update_count += 1
                stage3_candidate_pair_counts.append(int(candidate_pair_count))
                stage3_candidate_losses.append(float(stage3_candidate_loss.detach().cpu()))
                stage3_candidate_grad_norms.append(float(candidate_grad_norm.detach().cpu()))
                candidate_delta_squared = sum(
                    float(torch.sum((parameter.detach() - before) ** 2).cpu())
                    for parameter, before in zip(
                        self.macro_candidate_parameters,
                        candidate_parameters_before,
                        strict=True,
                    )
                )
                stage3_candidate_parameter_deltas.append(
                    math.sqrt(candidate_delta_squared)
                )

        def parameters_changed(name: str, parameters: tuple[Any, ...]) -> bool:
            return any(
                not torch.equal(before, current.detach())
                for before, current in zip(before_parameters[name], parameters, strict=True)
            )

        for name, parameter in self.policy.named_parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError(f"Updated MAPPO parameter {name} is non-finite.")
        def parameter_delta_l2(name: str, parameters: tuple[Any, ...]) -> float:
            squared = sum(
                float(torch.sum((current.detach() - before) ** 2).cpu())
                for before, current in zip(before_parameters[name], parameters, strict=True)
            )
            return float(math.sqrt(squared))

        def ratio_statistics(ratio: Any) -> dict[str, float]:
            values = ratio.detach().cpu().numpy().astype(np.float64, copy=False)
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }

        with torch.no_grad():
            macro_new_log_probs, _, macro_new_node_log_probs, macro_entropy = (
                self._evaluate_macro_actions(
                    macro_observations, macro_features, macro_masks, macro_actions
                )
            )
            micro_new_log_probs, _, micro_new_head_log_probs, micro_entropy = (
                self._evaluate_micro_actions(
                    micro_observations,
                    micro_features,
                    micro_masks,
                    micro_actions,
                    micro_conditions,
                )
            )
        macro_log_drift = macro_new_log_probs - macro_old_log_probs
        micro_log_drift = micro_new_log_probs - micro_old_log_probs
        macro_log_ratio = macro_new_log_probs - macro_old_log_probs
        micro_log_ratio = micro_new_log_probs - micro_old_log_probs
        macro_ratio = torch.exp(macro_log_ratio)
        micro_ratio = torch.exp(micro_log_ratio)
        macro_per_node_drift = macro_new_node_log_probs - macro_old_node_log_probs
        per_node_logprob_drift = {
            str(node_index): {
                "mean": float(torch.mean(macro_per_node_drift[:, node_index]).cpu()),
                "mean_abs": float(torch.mean(torch.abs(macro_per_node_drift[:, node_index])).cpu()),
                "max_abs": float(torch.max(torch.abs(macro_per_node_drift[:, node_index])).cpu()),
            }
            for node_index in range(9)
        }
        macro_approx_kl = torch.mean((macro_ratio - 1.0) - macro_log_ratio)
        micro_approx_kl = torch.mean((micro_ratio - 1.0) - micro_log_ratio)
        macro_clip_fraction = torch.mean(
            (torch.abs(macro_ratio - 1.0) > self.clip_epsilon).to(torch.float32)
        )
        micro_clip_fraction = torch.mean(
            (torch.abs(micro_ratio - 1.0) > self.clip_epsilon).to(torch.float32)
        )
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (
                macro_ratio,
                micro_ratio,
                macro_log_drift,
                micro_log_drift,
                macro_entropy,
                micro_entropy,
                macro_approx_kl,
                micro_approx_kl,
                macro_clip_fraction,
                micro_clip_fraction,
            )
        ):
            raise FloatingPointError("MAPPO post-update learning diagnostics are non-finite.")
        self.update_count += 1
        stage3_metrics: dict[str, Any] = {}
        if self.stage3_cache_value_config is not None:
            if self.cache_delta_q_value_head is None or stage3_tensors is None:
                raise RuntimeError("Stage3 update completed without its value-head tensors.")
            for name, parameter in self.cache_delta_q_value_head.named_parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    raise FloatingPointError(
                        f"Updated Stage3 ΔQ value-head parameter {name} is non-finite."
                    )
            with torch.no_grad():
                final_delta_q_predictions = self.cache_delta_q_value_head(
                    stage3_tensors["observations"],
                    stage3_tensors["node_features"],
                    stage3_tensors["candidate_features"],
                )
                valid_delta_q = stage3_tensors["update_masks"]
                if bool(torch.any(valid_delta_q).item()):
                    delta_q_rmse: float | None = float(
                        torch.sqrt(
                            torch.mean(
                                (
                                    final_delta_q_predictions[valid_delta_q]
                                    - stage3_tensors["delta_q_targets"][valid_delta_q]
                                )
                                ** 2
                            )
                        ).cpu()
                    )
                else:
                    delta_q_rmse = None
            self.stage3_update_count += 1
            stage3_metrics = {
                "algorithm_mode": COUNTERFACTUAL_VALUE_SCHEMA,
                "stage3_update_index": int(self.stage3_update_count),
                "stage3_loss_coefficients": {
                    "trigger_ranking": float(
                        self.stage3_cache_value_config.trigger_ranking_coefficient
                    ),
                    "candidate_ranking": float(
                        self.stage3_cache_value_config.candidate_ranking_coefficient
                    ),
                    "delta_q_huber": float(
                        self.stage3_cache_value_config.delta_q_huber_coefficient
                    ),
                },
                "stage3_trigger_ranking_loss": float(
                    np.mean(stage3_trigger_losses) if stage3_trigger_losses else 0.0
                ),
                "stage3_candidate_ranking_loss": float(
                    np.mean(stage3_candidate_losses) if stage3_candidate_losses else 0.0
                ),
                "stage3_delta_q_huber_loss": float(
                    np.mean(stage3_value_losses) if stage3_value_losses else 0.0
                ),
                "stage3_macro_actor_total_loss": float(
                    np.mean(macro_actor_total_losses)
                    if macro_actor_total_losses
                    else 0.0
                ),
                "stage3_value_head_gradient_norm": float(
                    np.mean(stage3_value_grad_norms)
                    if stage3_value_grad_norms
                    else 0.0
                ),
                "stage3_trigger_comparison_count": int(
                    sum(stage3_trigger_comparison_counts)
                ),
                "stage3_trigger_valid_supervision_row_count": int(
                    counterfactual_summary.get("trigger_supervision_row_count", 0)
                ),
                "stage3_trigger_tie_excluded_row_count": int(
                    counterfactual_summary.get("trigger_tie_excluded_row_count", 0)
                ),
                "stage3_trigger_delta_q_positive_count": int(
                    counterfactual_summary.get("trigger_delta_q_positive_count", 0)
                ),
                "stage3_trigger_delta_q_negative_count": int(
                    counterfactual_summary.get("trigger_delta_q_negative_count", 0)
                ),
                "stage3_trigger_delta_q_near_zero_count": int(
                    counterfactual_summary.get("trigger_delta_q_near_zero_count", 0)
                ),
                "stage3_candidate_pair_count": int(sum(stage3_candidate_pair_counts)),
                "stage3_candidate_pair_total_count": int(
                    counterfactual_summary.get("candidate_pair_total_count", 0)
                ),
                "stage3_candidate_pair_tie_count": int(
                    counterfactual_summary.get("candidate_pair_tie_count", 0)
                ),
                "stage3_candidate_pair_valid_count": int(
                    counterfactual_summary.get("candidate_pair_valid_count", 0)
                ),
                "stage3_candidate_delta_q_positive_count": int(
                    counterfactual_summary.get("candidate_delta_q_positive_count", 0)
                ),
                "stage3_candidate_delta_q_negative_count": int(
                    counterfactual_summary.get("candidate_delta_q_negative_count", 0)
                ),
                "stage3_candidate_delta_q_near_zero_count": int(
                    counterfactual_summary.get("candidate_delta_q_near_zero_count", 0)
                ),
                "stage3_delta_q_huber_scalar_count": int(
                    sum(stage3_value_scalar_counts)
                ),
                "stage3_ppo_trigger_gradient_relation": (
                    stage3_ppo_trigger_gradient_relation
                ),
                "stage3_candidate_supervised_update_count": int(
                    len(stage3_candidate_grad_norms)
                ),
                "stage3_candidate_supervised_update_total": int(
                    self.candidate_supervised_update_count
                ),
                "stage3_candidate_actor_gradient_norm": float(
                    np.mean(stage3_candidate_grad_norms)
                    if stage3_candidate_grad_norms
                    else 0.0
                ),
                "stage3_candidate_actor_parameter_delta_l2": float(
                    np.mean(stage3_candidate_parameter_deltas)
                    if stage3_candidate_parameter_deltas
                    else 0.0
                ),
                "stage3_candidate_actor_parameters_updated": parameters_changed(
                    "macro_candidate", self.macro_candidate_parameters
                ),
                "stage3_delta_q_rmse": delta_q_rmse,
                "stage3_delta_q_value_head_parameters_updated": parameters_changed(
                    "cache_delta_q_value",
                    tuple(self.cache_delta_q_value_head.parameters()),
                ),
                "stage3_delta_q_value_head_parameter_delta_l2": parameter_delta_l2(
                    "cache_delta_q_value",
                    tuple(self.cache_delta_q_value_head.parameters()),
                ),
                "stage3_counterfactual": counterfactual_summary,
            }
        return {
            "baseline_schema": MAPPO_BASELINE_SCHEMA,
            "update_index": int(self.update_count),
            "epochs": int(epochs),
            "macro_transition_count": len(collected.macro),
            "micro_transition_count": len(collected.micro),
            "macro_behavior_log_prob_max_abs_error": macro_old_error,
            "macro_node_behavior_log_prob_max_abs_error": macro_node_old_error,
            "micro_behavior_log_prob_max_abs_error": micro_old_error,
            "micro_head_behavior_log_prob_max_abs_error": micro_head_old_error,
            "macro_advantage_mean": float(macro_adv_mean),
            "macro_advantage_std": float(macro_adv_std),
            "micro_advantage_mean": float(micro_adv_mean),
            "micro_advantage_std": float(micro_adv_std),
            "macro_actor_loss": float(np.mean(macro_actor_losses)),
            "micro_actor_loss": float(np.mean(micro_actor_losses)),
            "macro_critic_loss": float(np.mean(macro_critic_losses)),
            "micro_critic_loss": float(np.mean(micro_critic_losses)),
            "macro_actor_gradient_norm": float(np.mean(macro_actor_grad_norms)),
            "micro_actor_gradient_norm": float(np.mean(micro_actor_grad_norms)),
            "macro_critic_gradient_norm": float(np.mean(macro_critic_grad_norms)),
            "micro_critic_gradient_norm": float(np.mean(micro_critic_grad_norms)),
            "macro_approx_kl": float(macro_approx_kl.cpu()),
            "micro_approx_kl": float(micro_approx_kl.cpu()),
            "macro_clip_fraction": float(macro_clip_fraction.cpu()),
            "micro_clip_fraction": float(micro_clip_fraction.cpu()),
            "macro_entropy": float(torch.mean(macro_entropy).cpu()),
            "micro_entropy": float(torch.mean(micro_entropy).cpu()),
            "macro_logprob_drift_mean": float(torch.mean(macro_log_drift).cpu()),
            "macro_logprob_drift_mean_abs": float(torch.mean(torch.abs(macro_log_drift)).cpu()),
            "macro_logprob_drift_max_abs": float(torch.max(torch.abs(macro_log_drift)).cpu()),
            "micro_logprob_drift_mean": float(torch.mean(micro_log_drift).cpu()),
            "micro_logprob_drift_mean_abs": float(torch.mean(torch.abs(micro_log_drift)).cpu()),
            "micro_logprob_drift_max_abs": float(torch.max(torch.abs(micro_log_drift)).cpu()),
            "macro_ratio_distribution": ratio_statistics(macro_ratio),
            "micro_ratio_distribution": ratio_statistics(micro_ratio),
            "per_node_logprob_drift": per_node_logprob_drift,
            "cache_keep_rate": float(collected.summary["cache_keep_rate"]),
            "cache_update_rate": float(collected.summary["cache_update_rate"]),
            "cache_update_candidate_counts_by_node": collected.summary[
                "cache_update_candidate_counts_by_node"
            ],
            "uav_stay_rate": float(collected.summary["uav_stay_rate"]),
            "uav_move_rate": float(collected.summary["uav_move_rate"]),
            "uav_move_feasible": bool(collected.summary["uav_move_feasible"]),
            "uav_joint_action_counts": collected.summary["uav_joint_action_counts"],
            "rollout_environment": collected.summary,
            "macro_actor_parameters_updated": parameters_changed(
                "macro_actor", self.macro_actor_parameters
            ),
            "micro_actor_parameters_updated": parameters_changed(
                "micro_actor", self.micro_actor_parameters
            ),
            "macro_critic_parameters_updated": parameters_changed(
                "macro_critic", self.macro_critic_parameters
            ),
            "micro_critic_parameters_updated": parameters_changed(
                "micro_critic", self.micro_critic_parameters
            ),
            "macro_actor_parameter_delta_l2": parameter_delta_l2(
                "macro_actor", self.macro_actor_parameters
            ),
            "micro_actor_parameter_delta_l2": parameter_delta_l2(
                "micro_actor", self.micro_actor_parameters
            ),
            "macro_critic_parameter_delta_l2": parameter_delta_l2(
                "macro_critic", self.macro_critic_parameters
            ),
            "micro_critic_parameter_delta_l2": parameter_delta_l2(
                "micro_critic", self.micro_critic_parameters
            ),
            "all_updated_parameters_finite": True,
            "numerical_errors": 0,
            **stage3_metrics,
        }

    def _checkpoint_contract(self) -> dict[str, Any]:
        config_path = Path(str(self.context.cfg["_meta"]["config_path"]))
        contract = {
            "baseline_schema": MAPPO_BASELINE_SCHEMA,
            "observation_dim": int(self.context.observation_spec.observation_dim),
            "macro_node_count": len(self.context.action_spec.cache_node_ids),
            "macro_node_roles": tuple(
                str(value).upper() for value in self.context.cache_nodes["node_roles"]
            ),
            "macro_candidate_feature_dim": int(
                CACHE_ECONOMIC_BASE_MACRO_FEATURE_DIM + CACHE_ECONOMIC_FEATURE_DIM
            ),
            "micro_head_count": len(self.context.action_spec.uav_ids),
            "micro_candidate_feature_dim": self.uav_feature_dim,
            "macro_interval_slots": MACRO_INTERVAL_SLOTS,
            "operation_candidate_probability_contract": (
                "legal_group_logmeanexp_then_conditional_candidate_v1"
            ),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "config_sha256": project_config.raw_config_sha256(config_path),
        }
        if self.policy.separate_trigger_candidate:
            contract["macro_candidate_head_schema"] = (
                "separate_legacy_trigger_and_conditional_candidate_v1"
            )
        return contract

    def _stage3_checkpoint_contract(self) -> dict[str, Any]:
        if (
            self.stage3_cache_value_config is None
            or self.cache_delta_q_value_head is None
            or not self.policy.separate_trigger_candidate
        ):
            raise RuntimeError("Stage3 checkpoint contract requested while mode is disabled.")
        return {
            "algorithm_mode": COUNTERFACTUAL_VALUE_SCHEMA,
            "micro_common_random_contract": (
                "per_slot_per_ordered_uav_operation_and_candidate_uniforms_inverse_cdf_v1"
            ),
            "config": self.stage3_cache_value_config.contract(),
            "value_head_schema": "shared_global_node_candidate_mlp_v1",
            "macro_probability_schema": (
                "legacy_trigger_group_plus_independent_update_candidate_v1"
            ),
            "candidate_optimizer_ownership": (
                "macro_actor_adam_candidate_only_gradient_isolated_v1"
            ),
            "node_feature_dim": CACHE_NODE_VALUE_FEATURE_DIM,
            "value_head_parameter_shapes": {
                name: tuple(int(dimension) for dimension in parameter.shape)
                for name, parameter in self.cache_delta_q_value_head.named_parameters()
            },
        }

    def _migrate_macro_actor_optimizer_state(
        self, saved_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Extend legacy Macro Adam moments by copying them to the split head.

        The new conditional Candidate encoder/scorer begin as exact copies of
        their legacy Trigger-scoring counterparts, so their Adam moments are
        copied from those same source tensors. Unknown groups or parameter
        layouts are rejected rather than partially restored.
        """

        saved_groups = saved_state.get("param_groups")
        live_state = self.macro_actor_optimizer.state_dict()
        live_groups = live_state.get("param_groups")
        if (
            not isinstance(saved_groups, list)
            or len(saved_groups) != 1
            or not isinstance(live_groups, list)
            or len(live_groups) != 1
        ):
            raise ValueError("Legacy Macro Actor optimizer must contain exactly one parameter group.")
        old_ids = saved_groups[0].get("params")
        current_ids = live_groups[0].get("params")
        if not isinstance(old_ids, list) or not isinstance(current_ids, list):
            raise ValueError("Legacy Macro Actor optimizer parameter ids are malformed.")
        all_names = tuple(
            name
            for name, parameter in self.policy.named_parameters()
            if any(parameter is target for target in self.macro_actor_parameters)
        )
        old_names = tuple(
            name
            for name in all_names
            if not name.startswith(MACRO_CONDITIONAL_CANDIDATE_PREFIXES)
        )
        if len(old_ids) != len(old_names) or len(current_ids) != len(all_names):
            raise ValueError(
                "Legacy Macro Actor optimizer parameter order/count does not match the supported migration."
            )
        source_ids = dict(zip(old_names, old_ids, strict=True))
        source_state = saved_state.get("state")
        if not isinstance(source_state, dict):
            raise ValueError("Legacy Macro Actor optimizer moments are malformed.")
        migrated = copy.deepcopy(saved_state)
        migrated_moments: dict[Any, Any] = {}
        for target_id, target_name in zip(current_ids, all_names, strict=True):
            source_name = target_name
            if target_name.startswith("macro_conditional_candidate_encoder."):
                source_name = target_name.replace(
                    "macro_conditional_candidate_encoder.",
                    "macro_cache_candidate_encoder.",
                    1,
                )
            elif target_name.startswith("macro_conditional_cache_scorer."):
                source_name = target_name.replace(
                    "macro_conditional_cache_scorer.", "macro_cache_scorer.", 1
                )
            if source_name not in source_ids:
                raise ValueError(
                    f"Macro Actor optimizer migration source is missing: {source_name}."
                )
            source_id = source_ids[source_name]
            if source_id in source_state:
                migrated_moments[target_id] = copy.deepcopy(source_state[source_id])
        migrated["state"] = migrated_moments
        migrated["param_groups"][0]["params"] = list(current_ids)
        return migrated

    @staticmethod
    def _validate_optimizer_checkpoint_state(
        optimizer: Any,
        saved_state: dict[str, Any],
        *,
        optimizer_name: str,
    ) -> None:
        """Reject optimizer state with incompatible groups, hyperparameters or moments."""

        current_state = optimizer.state_dict()
        saved_groups = saved_state.get("param_groups")
        if not isinstance(saved_groups, list) or len(saved_groups) != len(
            current_state["param_groups"]
        ):
            raise ValueError(f"Checkpoint optimizer group count mismatch: {optimizer_name}.")
        saved_moments = saved_state.get("state")
        if not isinstance(saved_moments, dict):
            raise ValueError(f"Checkpoint optimizer state is malformed: {optimizer_name}.")
        stable_keys = (
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        )
        for group_index, (saved_group, current_group) in enumerate(
            zip(saved_groups, current_state["param_groups"], strict=True)
        ):
            saved_ids = saved_group.get("params")
            current_ids = current_group.get("params")
            if not isinstance(saved_ids, list) or len(saved_ids) != len(current_ids):
                raise ValueError(
                    f"Checkpoint optimizer parameter count mismatch: {optimizer_name}[{group_index}]."
                )
            for key in stable_keys:
                if key in current_group and saved_group.get(key) != current_group.get(key):
                    raise ValueError(
                        f"Checkpoint optimizer hyperparameter mismatch: {optimizer_name}.{key}."
                    )
            live_parameters = optimizer.param_groups[group_index]["params"]
            for saved_parameter_id, live_parameter in zip(
                saved_ids, live_parameters, strict=True
            ):
                parameter_state = saved_moments.get(saved_parameter_id, {})
                if not isinstance(parameter_state, dict):
                    raise ValueError(
                        f"Checkpoint optimizer moment entry is malformed: {optimizer_name}."
                    )
                for state_name, value in parameter_state.items():
                    if not torch.is_tensor(value):
                        continue
                    if state_name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and tuple(
                        value.shape
                    ) != tuple(live_parameter.shape):
                        raise ValueError(
                            "Checkpoint optimizer moment shape mismatch: "
                            f"{optimizer_name}.{state_name} "
                            f"{tuple(value.shape)} != {tuple(live_parameter.shape)}."
                        )

    @staticmethod
    def _checkpoint_file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a self-contained checkpoint without replacing an existing file."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage3_enabled = self.stage3_cache_value_config is not None
        if stage3_enabled and (
            self.cache_delta_q_value_head is None
            or self.cache_delta_q_value_optimizer is None
            or self.counterfactual_micro_rng is None
        ):
            raise RuntimeError(
                "Stage3 checkpoint cannot be saved without its value head, optimizer and Micro RNG."
            )
        progress = {"update_count": int(self.update_count), "seed": self.seed}
        optimizers = {
            "macro_actor": self.macro_actor_optimizer.state_dict(),
            "micro_actor": self.micro_actor_optimizer.state_dict(),
            "macro_critic": self.macro_critic_optimizer.state_dict(),
            "micro_critic": self.micro_critic_optimizer.state_dict(),
        }
        if stage3_enabled:
            progress.update(
                {
                    "stage3_update_count": int(self.stage3_update_count),
                    "candidate_supervised_update_count": int(
                        self.candidate_supervised_update_count
                    ),
                    "counterfactual_rollout_count": int(self.counterfactual_rollout_count),
                }
            )
            optimizers["cache_delta_q_value"] = (
                self.cache_delta_q_value_optimizer.state_dict()
            )
        random_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "trainer_generator": self.generator.get_state(),
        }
        if stage3_enabled:
            random_state["counterfactual_micro"] = (
                self.counterfactual_micro_rng.getstate()
            )
        payload = {
            "checkpoint_version": (
                MAPPO_STAGE3_CHECKPOINT_VERSION
                if stage3_enabled
                else MAPPO_CHECKPOINT_VERSION
            ),
            "baseline_schema": MAPPO_BASELINE_SCHEMA,
            "model_contract": self._checkpoint_contract(),
            "progress": progress,
            "policy_state_dict": self.policy.state_dict(),
            "optimizers": optimizers,
            "random_state": random_state,
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "checkpoint_unix_time": float(time.time()),
            },
            "metadata": {} if metadata is None else dict(metadata),
        }
        if stage3_enabled:
            payload["algorithm_mode"] = COUNTERFACTUAL_VALUE_SCHEMA
            payload["stage3_contract"] = self._stage3_checkpoint_contract()
            payload["stage3_initialization"] = {
                "mode": self.stage3_initialization_mode,
                "source_checkpoint_sha256": self.stage3_source_checkpoint_sha256,
            }
            payload["cache_delta_q_value_head_state_dict"] = (
                self.cache_delta_q_value_head.state_dict()
            )
        created = False
        try:
            with destination.open("xb") as stream:
                created = True
                torch.save(payload, stream)
        except Exception:
            if created and destination.exists():
                destination.unlink()
            raise
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        allow_v2_initialization: bool = False,
    ) -> dict[str, Any]:
        """Strictly restore current checkpoints or explicitly migrate legacy heads."""

        source = Path(path).expanduser().resolve()
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch releases before the weights_only argument.
            payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("MAPPO checkpoint payload must be a mapping.")
        checkpoint_version = int(payload.get("checkpoint_version", -1))
        stage3_enabled = self.stage3_cache_value_config is not None
        legacy_stage3 = checkpoint_version == MAPPO_STAGE3_LEGACY_CHECKPOINT_VERSION
        current_stage3 = checkpoint_version == MAPPO_STAGE3_CHECKPOINT_VERSION
        stage3_checkpoint = legacy_stage3 or current_stage3
        if checkpoint_version == MAPPO_CHECKPOINT_VERSION:
            if stage3_enabled and not allow_v2_initialization:
                raise ValueError(
                    "Loading a v2 checkpoint into Stage3 requires explicit allow_v2_initialization=True."
                )
        elif legacy_stage3 or current_stage3:
            if not stage3_enabled:
                raise ValueError("A Stage3 checkpoint requires Stage3 mode to be explicitly enabled.")
        else:
            raise ValueError("Unsupported MAPPO checkpoint version.")
        if payload.get("baseline_schema") != MAPPO_BASELINE_SCHEMA:
            raise ValueError("MAPPO checkpoint baseline schema mismatch.")
        expected_contract = self._checkpoint_contract()
        saved_contract = payload.get("model_contract")
        expected_legacy_contract = dict(expected_contract)
        expected_legacy_contract.pop("macro_candidate_head_schema", None)
        if saved_contract not in (expected_contract, expected_legacy_contract):
            raise ValueError("MAPPO checkpoint model/config contract mismatch.")
        if stage3_checkpoint:
            if payload.get("algorithm_mode") != COUNTERFACTUAL_VALUE_SCHEMA:
                raise ValueError("Stage3 checkpoint algorithm mode mismatch.")
            expected_stage3_contract = self._stage3_checkpoint_contract()
            expected_legacy_stage3_contract = dict(expected_stage3_contract)
            expected_legacy_stage3_contract.pop("macro_probability_schema")
            expected_legacy_stage3_contract.pop("candidate_optimizer_ownership")
            accepted_stage3_contracts = (
                (expected_stage3_contract,)
                if current_stage3
                else (expected_legacy_stage3_contract,)
            )
            if payload.get("stage3_contract") not in accepted_stage3_contracts:
                raise ValueError("Stage3 checkpoint value-head/loss contract mismatch.")
        elif payload.get("stage3_contract") is not None:
            raise ValueError("v2 checkpoint unexpectedly contains a Stage3 contract.")
        saved_policy_state = payload.get("policy_state_dict")
        live_policy_state = self.policy.state_dict()
        conditional_policy_names = {
            name
            for name in live_policy_state
            if name.startswith(MACRO_CONDITIONAL_CANDIDATE_PREFIXES)
        }
        legacy_policy_names = set(live_policy_state) - conditional_policy_names
        split_policy_state = isinstance(saved_policy_state, dict) and set(
            saved_policy_state
        ) == set(live_policy_state)
        legacy_policy_state = bool(conditional_policy_names) and isinstance(
            saved_policy_state, dict
        ) and set(saved_policy_state) == legacy_policy_names
        if not split_policy_state and not legacy_policy_state:
            raise ValueError("MAPPO checkpoint policy parameter names differ from both supported schemas.")
        if current_stage3 and not split_policy_state:
            raise ValueError("Current Stage3 checkpoint is missing conditional Candidate parameters.")
        if legacy_stage3 and (
            saved_contract != expected_legacy_contract or not legacy_policy_state
        ):
            raise ValueError("Legacy Stage3 checkpoint schema and policy tensors are inconsistent.")
        if saved_contract == expected_contract and not split_policy_state:
            raise ValueError("Checkpoint declares the split Candidate schema but omits its parameters.")
        if (
            conditional_policy_names
            and saved_contract == expected_legacy_contract
            and not legacy_policy_state
        ):
            raise ValueError("Legacy model contract does not match its policy parameter names.")
        for name in (live_policy_state if split_policy_state else legacy_policy_names):
            live_tensor = live_policy_state[name]
            saved_tensor = saved_policy_state[name]
            if not torch.is_tensor(saved_tensor) or tuple(saved_tensor.shape) != tuple(
                live_tensor.shape
            ):
                raise ValueError(f"MAPPO checkpoint policy tensor shape mismatch: {name}.")
        policy_needs_migration = bool(conditional_policy_names) and legacy_policy_state
        if policy_needs_migration:
            migrated_policy_state = copy.deepcopy(saved_policy_state)
            for target_name in sorted(conditional_policy_names):
                if target_name.startswith("macro_conditional_candidate_encoder."):
                    source_name = target_name.replace(
                        "macro_conditional_candidate_encoder.",
                        "macro_cache_candidate_encoder.",
                        1,
                    )
                else:
                    source_name = target_name.replace(
                        "macro_conditional_cache_scorer.", "macro_cache_scorer.", 1
                    )
                if source_name not in saved_policy_state:
                    raise ValueError(
                        f"Legacy Candidate migration source parameter is missing: {source_name}."
                    )
                migrated_tensor = saved_policy_state[source_name]
                if tuple(migrated_tensor.shape) != tuple(live_policy_state[target_name].shape):
                    raise ValueError(
                        f"Legacy Candidate migration shape mismatch: {source_name} -> {target_name}."
                    )
                migrated_policy_state[target_name] = migrated_tensor.detach().clone()
            saved_policy_state = migrated_policy_state
        saved_stage3_head_state = None
        stage3_initialization: dict[str, Any] | None = None
        if stage3_checkpoint:
            if self.cache_delta_q_value_head is None:
                raise RuntimeError("Stage3 checkpoint restore has no initialized ΔQ head.")
            saved_stage3_head_state = payload.get("cache_delta_q_value_head_state_dict")
            live_stage3_head_state = self.cache_delta_q_value_head.state_dict()
            if not isinstance(saved_stage3_head_state, dict) or set(
                saved_stage3_head_state
            ) != set(live_stage3_head_state):
                raise ValueError("Stage3 checkpoint value-head parameter names are incompatible.")
            for name, live_tensor in live_stage3_head_state.items():
                saved_tensor = saved_stage3_head_state[name]
                if not torch.is_tensor(saved_tensor) or tuple(saved_tensor.shape) != tuple(
                    live_tensor.shape
                ):
                    raise ValueError(f"Stage3 checkpoint value-head tensor mismatch: {name}.")
            initialization_payload = payload.get("stage3_initialization")
            if (
                not isinstance(initialization_payload, dict)
                or not isinstance(initialization_payload.get("mode"), str)
                or not initialization_payload["mode"]
            ):
                raise ValueError("Stage3 checkpoint initialization provenance is missing.")
            source_checkpoint_hash = initialization_payload.get(
                "source_checkpoint_sha256"
            )
            if source_checkpoint_hash is not None and (
                not isinstance(source_checkpoint_hash, str)
                or len(source_checkpoint_hash) != 64
                or any(value not in "0123456789abcdef" for value in source_checkpoint_hash)
            ):
                raise ValueError("Stage3 source checkpoint SHA-256 metadata is malformed.")
            if (
                initialization_payload["mode"].startswith("migrated_v2")
                and source_checkpoint_hash is None
            ):
                raise ValueError("Migrated Stage3 checkpoint is missing its v2 source hash.")
            stage3_initialization = {
                "mode": initialization_payload["mode"],
                "source_checkpoint_sha256": source_checkpoint_hash,
            }
        optimizer_states = payload["optimizers"]
        expected_base_optimizers = {
            "macro_actor",
            "micro_actor",
            "macro_critic",
            "micro_critic",
        }
        if not isinstance(optimizer_states, dict) or not expected_base_optimizers.issubset(
            optimizer_states
        ):
            raise ValueError("MAPPO checkpoint is missing one or more base optimizer states.")
        if checkpoint_version == MAPPO_CHECKPOINT_VERSION and set(optimizer_states) != expected_base_optimizers:
            raise ValueError("v2 MAPPO checkpoint optimizer set is not the exact legacy set.")
        if stage3_checkpoint and set(
            optimizer_states
        ) != expected_base_optimizers | {"cache_delta_q_value"}:
            raise ValueError("Stage3 checkpoint optimizer set is incomplete or unexpected.")
        live_optimizers = {
            "macro_actor": self.macro_actor_optimizer,
            "micro_actor": self.micro_actor_optimizer,
            "macro_critic": self.macro_critic_optimizer,
            "micro_critic": self.micro_critic_optimizer,
        }
        if stage3_checkpoint:
            if self.cache_delta_q_value_optimizer is None:
                raise RuntimeError("Stage3 optimizer was not initialized before checkpoint loading.")
            live_optimizers["cache_delta_q_value"] = self.cache_delta_q_value_optimizer
        optimizer_states_to_load = dict(optimizer_states)
        macro_optimizer_migrated = policy_needs_migration
        if macro_optimizer_migrated:
            optimizer_states_to_load["macro_actor"] = (
                self._migrate_macro_actor_optimizer_state(
                    optimizer_states["macro_actor"]
                )
            )
        for optimizer_name, optimizer in live_optimizers.items():
            self._validate_optimizer_checkpoint_state(
                optimizer,
                optimizer_states_to_load[optimizer_name],
                optimizer_name=optimizer_name,
            )
        progress = payload.get("progress")
        if not isinstance(progress, dict) or "update_count" not in progress:
            raise ValueError("MAPPO checkpoint progress metadata is missing.")
        update_count = int(progress["update_count"])
        if update_count < 0:
            raise ValueError("MAPPO checkpoint update count cannot be negative.")
        if stage3_enabled and progress.get("seed") is None:
            raise ValueError("Stage3 restore/migration requires a recoverable trainer seed.")
        stage3_update_count = 0
        candidate_supervised_update_count = 0
        counterfactual_rollout_count = 0
        if stage3_checkpoint:
            stage3_update_count = int(progress.get("stage3_update_count", -1))
            counterfactual_rollout_count = int(
                progress.get("counterfactual_rollout_count", -1)
            )
            if stage3_update_count < 0 or counterfactual_rollout_count < 0:
                raise ValueError("Stage3 checkpoint progress counters are missing or invalid.")
            if current_stage3:
                candidate_supervised_update_count = int(
                    progress.get("candidate_supervised_update_count", -1)
                )
                if candidate_supervised_update_count < 0:
                    raise ValueError("Stage3 Candidate update counter is missing or invalid.")
        random_state = payload.get("random_state")
        required_rng = {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "trainer_generator",
        }
        if stage3_checkpoint:
            required_rng.add("counterfactual_micro")
        if not isinstance(random_state, dict) or not required_rng.issubset(random_state):
            raise ValueError("MAPPO checkpoint is missing recoverable Python/NumPy/Torch RNG state.")
        if not torch.is_tensor(random_state["torch_cpu"]) or not torch.is_tensor(
            random_state["trainer_generator"]
        ):
            raise ValueError("MAPPO checkpoint Torch RNG state tensors are malformed.")
        saved_counterfactual_rng_state = None
        if stage3_checkpoint:
            if not stage3_enabled or self.counterfactual_micro_rng is None:
                raise RuntimeError("Stage3 Micro RNG stream was not initialized before restore.")
            saved_counterfactual_rng_state = random_state["counterfactual_micro"]
            try:
                random.Random().setstate(saved_counterfactual_rng_state)
            except Exception as exc:
                raise ValueError(
                    "Stage3 checkpoint Micro common-random state is malformed."
                ) from exc
        elif stage3_enabled and checkpoint_version == MAPPO_CHECKPOINT_VERSION:
            try:
                progress_seed = int(progress["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Stage3 v2 migration has no valid source trainer seed.") from exc
            if progress_seed != progress["seed"]:
                raise ValueError("Stage3 v2 migration trainer seed is not an integer value.")

        # Mutate the live trainer only after every schema, shape, optimizer and RNG check passes.
        self.policy.load_state_dict(saved_policy_state, strict=True)
        self.macro_actor_optimizer.load_state_dict(optimizer_states_to_load["macro_actor"])
        self.micro_actor_optimizer.load_state_dict(optimizer_states_to_load["micro_actor"])
        self.macro_critic_optimizer.load_state_dict(optimizer_states_to_load["macro_critic"])
        self.micro_critic_optimizer.load_state_dict(optimizer_states_to_load["micro_critic"])
        self.update_count = update_count
        self.seed = progress.get("seed")
        if stage3_checkpoint:
            if (
                self.cache_delta_q_value_head is None
                or self.cache_delta_q_value_optimizer is None
                or saved_stage3_head_state is None
            ):
                raise RuntimeError("Stage3 checkpoint restore has no initialized ΔQ head.")
            self.cache_delta_q_value_head.load_state_dict(
                saved_stage3_head_state, strict=True
            )
            self.cache_delta_q_value_optimizer.load_state_dict(
                optimizer_states_to_load["cache_delta_q_value"]
            )
            self.stage3_update_count = stage3_update_count
            self.candidate_supervised_update_count = candidate_supervised_update_count
            self.counterfactual_rollout_count = counterfactual_rollout_count
            if stage3_initialization is None:
                raise RuntimeError("Stage3 initialization provenance was not validated.")
            self.stage3_initialization_mode = stage3_initialization["mode"]
            self.stage3_source_checkpoint_sha256 = stage3_initialization[
                "source_checkpoint_sha256"
            ]
            self.counterfactual_micro_rng.setstate(saved_counterfactual_rng_state)
        elif stage3_enabled:
            self.stage3_update_count = 0
            self.candidate_supervised_update_count = 0
            self.counterfactual_rollout_count = 0
            self.stage3_initialization_mode = (
                "migrated_v2_policy_and_optimizer_moments_split_candidate_fresh_stage3_delta_q_head"
            )
            self.stage3_source_checkpoint_sha256 = self._checkpoint_file_sha256(source)
            self.counterfactual_micro_rng = random.Random(
                (int(self.seed) ^ 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF
            )
        else:
            self.stage3_update_count = 0
            self.candidate_supervised_update_count = 0
            self.counterfactual_rollout_count = 0
            self.stage3_initialization_mode = None
            self.stage3_source_checkpoint_sha256 = None
            self.counterfactual_micro_rng = None
        random.setstate(random_state["python"])
        np.random.set_state(random_state["numpy"])
        torch.set_rng_state(random_state["torch_cpu"].cpu())
        if torch.cuda.is_available() and random_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(random_state["torch_cuda"])
        self.generator.set_state(random_state["trainer_generator"].cpu())
        restored = {
            "update_count": self.update_count,
            "seed": self.seed,
            "metadata": dict(payload.get("metadata", {})),
            "model_contract": expected_contract,
            "checkpoint_version": checkpoint_version,
            "conditional_candidate_policy_migrated": policy_needs_migration,
            "macro_actor_optimizer_migrated": macro_optimizer_migrated,
        }
        if stage3_enabled:
            restored.update(
                {
                    "algorithm_mode": COUNTERFACTUAL_VALUE_SCHEMA,
                    "stage3_update_count": int(self.stage3_update_count),
                    "candidate_supervised_update_count": int(
                        self.candidate_supervised_update_count
                    ),
                    "counterfactual_rollout_count": int(
                        self.counterfactual_rollout_count
                    ),
                    "v2_initialization_migrated": (
                        checkpoint_version == MAPPO_CHECKPOINT_VERSION
                    ),
                }
            )
        return restored
