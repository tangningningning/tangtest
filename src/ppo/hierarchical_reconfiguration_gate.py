# -*- coding: utf-8 -*-
"""Hierarchical ``WHEN`` gate for the configuration-trigger policy.

The module is deliberately additive to the existing V3 Direct-CF path.  The
existing dynamic-trigger Actor remains the ``WHAT`` policy; this module only
adds an independent binary policy deciding whether the configuration branch
is entered at all.

The public APIs are split into three small pieces:

* :class:`ReconfigurationGate` and
  :class:`HierarchicalReconfigurationGatePolicy` implement the policy
  factorisation and the strict ``g=0 -> all-zero`` action mapping.
* :func:`build_gate_counterfactual_sample` and
  :func:`update_gate_on_counterfactual_batch` implement the real-return
  Joint-Null credit and the one-epoch clipped PPO update.
* The serializer helpers provide the fields needed to reconstruct an
  independent Gate update.  The causal-artifact module uses these helpers for
  a separate, versioned artifact schema; historical Direct-CF artifacts are
  not changed.

No environment, configuration solver, Matching, KKT, or training loop is
called here.  A caller supplies the already executed CONFIG/NULL branch
returns, which keeps fixture validation and future formal rollout integration
separate from the policy mathematics.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ppo import ppo_trigger_policy

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - import-only environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


HIERARCHICAL_GATE_METHOD = "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift"
HIERARCHICAL_GATE_ALGORITHM_SCHEMA = (
    "proposed_hierarchical_reconfiguration_gate_direct_cf_refdrift_v1"
)
HIERARCHICAL_GATE_ARTIFACT_SCHEMA = "hierarchical_reconfiguration_gate_artifact_v1"
HIERARCHICAL_GATE_ARCHITECTURE_ID = "hierarchical_reconfiguration_gate_v1"
HIERARCHICAL_GATE_ARCHITECTURE_VERSION = 1
GATE_ACTION_CONFIGURE = 1
GATE_ACTION_NULL = 0
DEFAULT_GATE_HIDDEN_SIZES = (32,)
DEFAULT_GATE_CLIP_EPSILON = 0.2
DEFAULT_GATE_UPDATE_EPOCHS = 1


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("hierarchical_reconfiguration_gate 需要 PyTorch。")


def _policy_device(policy: Any) -> Any:
    _require_torch()
    try:
        return next(policy.parameters()).device
    except StopIteration as exc:
        raise ValueError("policy 必须包含至少一个参数。") from exc


def _as_finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是finite数值。")
    return number


def _binary_action(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是0或1。")
    action = int(value)
    if action != value or action not in {0, 1}:
        raise ValueError(f"{name} 必须是0或1。")
    return action


def _clone_state(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _capture_global_rng_state() -> dict[str, Any]:
    """Capture policy/fixture RNGs without drawing from any of them."""

    _require_torch()
    return {
        "python_random_state": copy.deepcopy(random.getstate()),
        "numpy_random_state": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state_all": (
            None
            if not torch.cuda.is_available()
            else [state.clone() for state in torch.cuda.get_rng_state_all()]
        ),
    }


def _restore_global_rng_state(state: dict[str, Any]) -> None:
    _require_torch()
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_state_all")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Joint-Null RNG state需要CUDA，但当前CUDA不可用。")
        torch.cuda.set_rng_state_all(cuda_states)


@dataclass(frozen=True)
class GateActionSample:
    """One binary Gate draw, including the old-policy probability/log-prob."""

    action: int
    probability_one: float
    log_prob: float
    entropy: float
    logit: float


@dataclass(frozen=True)
class GateEvaluation:
    """Batched Gate distribution evaluation used by the PPO loss."""

    logits: Any
    probability_one: Any
    probability_table: Any
    per_action_log_probs: Any
    chosen_log_prob: Any
    entropy: Any


class ReconfigurationGate(nn.Module if nn is not None else object):
    """Small, independent binary policy for ``g_t = configure?``.

    The default PyTorch Linear initialisation is intentionally retained.  It
    is a neutral, standard initialisation and does not encode an always-zero
    or always-one prior.  The Gate receives exactly the existing configuration
    observation; no time, cost, demand, or other new feature is introduced.
    """

    architecture_id = HIERARCHICAL_GATE_ARCHITECTURE_ID
    architecture_version = HIERARCHICAL_GATE_ARCHITECTURE_VERSION

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: tuple[int, ...] = DEFAULT_GATE_HIDDEN_SIZES,
    ) -> None:
        _require_torch()
        super().__init__()
        if int(observation_dim) <= 0:
            raise ValueError("Gate observation_dim 必须大于0。")
        sizes = tuple(int(size) for size in hidden_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("Gate hidden_sizes 必须是非空正整数序列。")
        self.observation_dim = int(observation_dim)
        self.hidden_sizes = sizes
        layers: list[Any] = []
        last_dim = self.observation_dim
        for size in sizes:
            layers.extend([nn.Linear(last_dim, size), nn.Tanh()])
            last_dim = size
        self.gate_encoder = nn.Sequential(*layers)
        self.gate_logit_head = nn.Linear(last_dim, 1)

    def _as_batch(self, observations: Any) -> tuple[Any, bool]:
        _require_torch()
        device = _policy_device(self)
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        single = False
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            single = True
        if obs.ndim != 2 or tuple(obs.shape[1:]) != (self.observation_dim,):
            raise ValueError(
                f"Gate observations shape={tuple(obs.shape)}，期望 [D] 或 "
                f"[B,{self.observation_dim}]。"
            )
        if not bool(torch.all(torch.isfinite(obs)).item()):
            raise ValueError("Gate observations 包含NaN/Inf。")
        return obs, single

    def forward(self, observations: Any) -> Any:
        obs, _ = self._as_batch(observations)
        return self.gate_logit_head(self.gate_encoder(obs)).squeeze(-1)

    def _distribution(self, observations: Any) -> tuple[Any, Any]:
        obs, _ = self._as_batch(observations)
        logits = self.gate_logit_head(self.gate_encoder(obs)).squeeze(-1)
        categorical_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        return logits, torch.distributions.Categorical(logits=categorical_logits)

    def evaluate_actions(self, observations: Any, gate_actions: Any) -> GateEvaluation:
        """Evaluate a batch of explicit binary Gate actions."""

        logits, distribution = self._distribution(observations)
        actions = torch.as_tensor(
            gate_actions,
            dtype=torch.long,
            device=logits.device,
        ).reshape(-1)
        if actions.shape != (logits.shape[0],):
            raise ValueError(
                f"gate_actions shape={tuple(actions.shape)}，期望 {(logits.shape[0],)}。"
            )
        if not bool(torch.all((actions == 0) | (actions == 1)).item()):
            raise ValueError("gate_actions 必须是0/1。")
        probability_table = distribution.probs
        per_action_log_probs = torch.log(probability_table)
        chosen_log_prob = distribution.log_prob(actions)
        return GateEvaluation(
            logits=logits,
            probability_one=probability_table[:, 1],
            probability_table=probability_table,
            per_action_log_probs=per_action_log_probs,
            chosen_log_prob=chosen_log_prob,
            entropy=distribution.entropy(),
        )

    def sample_action(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
        force_action: int | None = None,
    ) -> GateActionSample:
        """Draw one Gate action.

        A stochastic forced draw still performs the underlying distribution
        draw before replacing its value.  CONFIG and NULL Joint-Null branches
        therefore consume the same Gate draw and can restore the same branch
        RNG checkpoint without changing the caller's RNG after return.
        """

        logits, distribution = self._distribution(observation)
        if logits.shape != (1,):
            raise ValueError("Gate sample_action 只接受单个 observation。")
        if force_action is not None:
            forced = _binary_action(force_action, "force_action")
        else:
            forced = None
        if deterministic:
            sampled = torch.argmax(distribution.probs, dim=-1)
        else:
            sampled = distribution.sample()
        action_tensor = sampled if forced is None else torch.full_like(sampled, forced)
        action = int(action_tensor.item())
        table = distribution.probs
        return GateActionSample(
            action=action,
            probability_one=float(table[0, 1].detach().cpu()),
            log_prob=float(distribution.log_prob(action_tensor)[0].detach().cpu()),
            entropy=float(distribution.entropy()[0].detach().cpu()),
            logit=float(logits[0].detach().cpu()),
        )


@dataclass(frozen=True)
class HierarchicalActionSample:
    """Compatibility-shaped action sample plus explicit Gate fields."""

    action_vector: np.ndarray
    cache_triggers: np.ndarray
    uav_triggers: dict[str, int]
    joint_log_prob: float
    conditional_entropy: float
    value: float
    per_action_probabilities: np.ndarray
    per_action_probability_table: np.ndarray
    per_action_log_probs: np.ndarray
    per_action_logits: np.ndarray
    per_action_masks: np.ndarray
    gate_action: int
    p_old_gate: float
    gate_old_logprob: float
    gate_entropy: float
    gate_logit: float
    subpolicy_action_vector: np.ndarray


@dataclass(frozen=True)
class HierarchicalEvaluation:
    gate: GateEvaluation
    subpolicy: ppo_trigger_policy.TriggerEvaluation
    joint_log_prob: Any
    value: Any


def all_zero_configuration_action(action_dim: int) -> np.ndarray:
    if isinstance(action_dim, bool) or int(action_dim) <= 0:
        raise ValueError("action_dim 必须是正整数。")
    return np.zeros(int(action_dim), dtype=np.int8)


def finalize_configuration_action(
    gate_action: Any,
    subpolicy_action: Any,
    action_dim: int,
) -> np.ndarray:
    """Map ``g`` and the WHAT action to the environment-visible 8-bit action."""

    gate = _binary_action(gate_action, "gate_action")
    vector = np.asarray(subpolicy_action, dtype=np.int8).reshape(-1)
    if vector.shape != (int(action_dim),) or not np.all(np.isin(vector, [0, 1])):
        raise ValueError("subpolicy_action 必须是指定长度的0/1向量。")
    if gate == GATE_ACTION_NULL:
        return all_zero_configuration_action(int(action_dim))
    return vector.copy()


class HierarchicalReconfigurationGatePolicy(nn.Module if nn is not None else object):
    """Independent Gate + existing full shared autoregressive WHAT Actor."""

    architecture_id = HIERARCHICAL_GATE_METHOD
    architecture_version = HIERARCHICAL_GATE_ARCHITECTURE_VERSION

    def __init__(
        self,
        sub_policy: Any,
        gate: ReconfigurationGate,
    ) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(gate, ReconfigurationGate):
            raise TypeError("gate 必须是 ReconfigurationGate。")
        if not hasattr(sub_policy, "sample_action") or not hasattr(sub_policy, "evaluate_actions"):
            raise TypeError("sub_policy 必须是现有 Trigger Actor。")
        if int(getattr(sub_policy, "observation_dim")) != int(gate.observation_dim):
            raise ValueError("Gate 与 WHAT Actor 必须使用相同 observation_dim。")
        self.sub_policy = sub_policy
        self.gate = gate

    @property
    def observation_dim(self) -> int:
        return int(self.sub_policy.observation_dim)

    @property
    def action_spec(self) -> ppo_trigger_policy.TriggerActionSpec:
        return self.sub_policy.action_spec

    @property
    def action_dim(self) -> int:
        return int(self.sub_policy.action_dim)

    @property
    def autoregressive_hidden_size(self) -> int:
        return int(self.sub_policy.autoregressive_hidden_size)

    def evaluate_gate(self, observations: Any, gate_actions: Any) -> GateEvaluation:
        return self.gate.evaluate_actions(observations, gate_actions)

    def evaluate_gate_actions(self, observations: Any, gate_actions: Any) -> GateEvaluation:
        """Named adapter used by :func:`ppo_update.compute_gate_direct_cf_actor_loss`."""

        return self.gate.evaluate_actions(observations, gate_actions)

    def evaluate_actions(
        self,
        observations: Any,
        action_vectors: Any,
        action_masks: Any = None,
        gate_actions: Any = None,
    ) -> HierarchicalEvaluation:
        """Evaluate Gate and WHAT distributions separately.

        ``gate_actions`` should always be supplied by a hierarchical rollout.
        For compatibility-only read paths it may be omitted; then a nonzero
        final action is treated as CONFIG and an all-zero action as NULL.  The
        latter is intentionally not used for Gate training because CONFIG can
        legitimately sample an all-zero WHAT vector.
        """

        actions = torch.as_tensor(action_vectors, dtype=torch.float32, device=_policy_device(self))
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2 or tuple(actions.shape[1:]) != (self.action_dim,):
            raise ValueError("hierarchical action_vectors shape非法。")
        if gate_actions is None:
            inferred = torch.any(actions != 0.0, dim=1).to(torch.long)
        else:
            inferred = torch.as_tensor(gate_actions, dtype=torch.long, device=actions.device).reshape(-1)
        gate_eval = self.gate.evaluate_actions(
            torch.as_tensor(observations, dtype=torch.float32, device=actions.device),
            inferred,
        )
        sub_eval = self.sub_policy.evaluate_actions(
            observations,
            actions,
            action_masks=action_masks,
        )
        sub_joint = sub_eval.joint_log_prob
        if sub_joint.ndim == 0:
            sub_joint = sub_joint.unsqueeze(0)
        return HierarchicalEvaluation(
            gate=gate_eval,
            subpolicy=sub_eval,
            joint_log_prob=gate_eval.chosen_log_prob + sub_joint,
            value=sub_eval.value,
        )

    def conditional_hidden_states(self, observations: Any, action_vectors: Any) -> Any:
        return self.sub_policy.conditional_hidden_states(observations, action_vectors)

    def _zero_subpolicy_sample(self, observation: Any) -> ppo_trigger_policy.TriggerActionSample:
        """Build a no-RNG all-zero WHAT sample for ordinary NULL decisions."""

        zeros = all_zero_configuration_action(self.action_dim)
        masks = np.ones((self.action_dim, 2), dtype=bool)
        evaluation = self.sub_policy.evaluate_actions(
            observation,
            zeros,
            action_masks=masks,
        )
        return ppo_trigger_policy.TriggerActionSample(
            action_vector=zeros.copy(),
            cache_triggers=self.action_spec.action_vector_to_triggers(zeros)[0],
            uav_triggers=self.action_spec.action_vector_to_triggers(zeros)[1],
            joint_log_prob=float(evaluation.joint_log_prob.detach().cpu()),
            conditional_entropy=float(evaluation.entropy.detach().cpu()),
            value=float(evaluation.value.detach().cpu()),
            per_action_probabilities=evaluation.per_action_probabilities.detach().cpu().numpy().astype(np.float64),
            per_action_probability_table=evaluation.per_action_probability_table.detach().cpu().numpy().astype(np.float64),
            per_action_log_probs=evaluation.per_action_log_probs.detach().cpu().numpy().astype(np.float64),
            per_action_logits=evaluation.per_action_logits.detach().cpu().numpy().astype(np.float64),
            per_action_masks=masks,
        )

    def sample_action(
        self,
        observation: Any,
        deterministic: bool = False,
        action_mask: Callable[[int, np.ndarray], Any] | Any = None,
        force_gate_action: int | None = None,
        consume_subpolicy_draw_for_null: bool = False,
        **subpolicy_kwargs: Any,
    ) -> HierarchicalActionSample:
        """Sample ``g`` and then the existing WHAT policy.

        ``consume_subpolicy_draw_for_null=True`` is used by Joint-Null CF so
        CONFIG and NULL branches consume the same number of old-policy draws;
        it does not change the NULL action exposed to the environment.
        """

        gate_sample = self.gate.sample_action(
            observation,
            deterministic=deterministic,
            force_action=force_gate_action,
        )
        if gate_sample.action == GATE_ACTION_CONFIGURE:
            sub_sample = self.sub_policy.sample_action(
                observation,
                deterministic=deterministic,
                action_mask=action_mask,
                **subpolicy_kwargs,
            )
        elif consume_subpolicy_draw_for_null:
            sub_sample = self.sub_policy.sample_action(
                observation,
                deterministic=False,
                action_mask=None,
                fixed_action_prefix=all_zero_configuration_action(self.action_dim),
            )
        else:
            sub_sample = self._zero_subpolicy_sample(observation)

        action_vector = finalize_configuration_action(
            gate_sample.action,
            sub_sample.action_vector,
            self.action_dim,
        )
        cache_triggers, uav_triggers = self.action_spec.action_vector_to_triggers(action_vector)
        return HierarchicalActionSample(
            action_vector=action_vector,
            cache_triggers=cache_triggers,
            uav_triggers=uav_triggers,
            joint_log_prob=float(gate_sample.log_prob + sub_sample.joint_log_prob),
            conditional_entropy=float(gate_sample.entropy + sub_sample.conditional_entropy),
            value=float(sub_sample.value),
            per_action_probabilities=np.asarray(sub_sample.per_action_probabilities, dtype=np.float64).copy(),
            per_action_probability_table=np.asarray(sub_sample.per_action_probability_table, dtype=np.float64).copy(),
            per_action_log_probs=np.asarray(sub_sample.per_action_log_probs, dtype=np.float64).copy(),
            per_action_logits=np.asarray(sub_sample.per_action_logits, dtype=np.float64).copy(),
            per_action_masks=np.asarray(sub_sample.per_action_masks, dtype=bool).copy(),
            gate_action=int(gate_sample.action),
            p_old_gate=float(gate_sample.probability_one),
            gate_old_logprob=float(gate_sample.log_prob),
            gate_entropy=float(gate_sample.entropy),
            gate_logit=float(gate_sample.logit),
            subpolicy_action_vector=np.asarray(sub_sample.action_vector, dtype=np.int8).copy(),
        )


def build_hierarchical_policy(
    observation_dim: int,
    action_spec: ppo_trigger_policy.TriggerActionSpec,
    *,
    subpolicy_hidden_sizes: tuple[int, ...] = (128, 128),
    autoregressive_hidden_size: int = 64,
    gate_hidden_sizes: tuple[int, ...] = DEFAULT_GATE_HIDDEN_SIZES,
) -> HierarchicalReconfigurationGatePolicy:
    """Construct a new hierarchical policy without changing observation shape."""

    sub_policy = ppo_trigger_policy.FullSharedActorCritic(
        observation_dim,
        action_spec,
        hidden_sizes=tuple(int(value) for value in subpolicy_hidden_sizes),
        autoregressive_hidden_size=int(autoregressive_hidden_size),
    )
    gate = ReconfigurationGate(
        observation_dim,
        hidden_sizes=tuple(int(value) for value in gate_hidden_sizes),
    )
    return HierarchicalReconfigurationGatePolicy(sub_policy, gate)


def hierarchical_policy_metadata(policy: HierarchicalReconfigurationGatePolicy) -> dict[str, Any]:
    if not isinstance(policy, HierarchicalReconfigurationGatePolicy):
        raise TypeError("policy 必须是 HierarchicalReconfigurationGatePolicy。")
    return {
        "method_id": HIERARCHICAL_GATE_METHOD,
        "algorithm_schema": HIERARCHICAL_GATE_ALGORITHM_SCHEMA,
        "architecture_id": HIERARCHICAL_GATE_ARCHITECTURE_ID,
        "architecture_version": HIERARCHICAL_GATE_ARCHITECTURE_VERSION,
        "observation_dim": int(policy.observation_dim),
        "gate": {
            "input": "existing_configuration_observation",
            "hidden_sizes": [int(value) for value in policy.gate.hidden_sizes],
            "output": "binary_g_t_configure_or_null",
            "optimizer": "independent_adam",
            "refdrift": "excluded",
            "entropy_bonus": False,
        },
        "what": {
            "architecture": ppo_trigger_policy.actor_architecture_metadata(policy.sub_policy),
            "semantics": "existing_full_shared_autoregressive_8_trigger_policy",
            "optimizer": "independent_existing_direct_cf_actor_optimizer",
        },
        "observation_unchanged": True,
        "environment_action": "all_zero_8_bit_action_when_gate_is_zero",
    }


@dataclass(frozen=True)
class GateBranchReturn:
    """Real environment return decomposition for one frozen branch."""

    branch: str
    return_g: float
    return_q: float
    return_cache: float
    return_uav: float
    environment_steps: int

    def __post_init__(self) -> None:
        if self.branch not in {"CONFIG", "NULL"}:
            raise ValueError("GateBranchReturn.branch 必须是CONFIG或NULL。")
        values = (self.return_g, self.return_q, self.return_cache, self.return_uav)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Gate branch return包含NaN/Inf。")
        expected = float(self.return_q) - float(self.return_cache) - float(self.return_uav)
        if not math.isclose(float(self.return_g), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Gate branch必须满足G=Q-cache-UAV。")
        if int(self.environment_steps) < 0:
            raise ValueError("Gate branch environment_steps不能为负。")


@dataclass(frozen=True)
class JointNullBranchResult:
    config_sample: HierarchicalActionSample
    null_sample: HierarchicalActionSample
    config_return: GateBranchReturn
    null_return: GateBranchReturn
    crn_metadata: dict[str, Any]


def run_joint_null_branches(
    policy: HierarchicalReconfigurationGatePolicy,
    observation: Any,
    continuation: Callable[[str, HierarchicalActionSample], GateBranchReturn],
    *,
    action_mask: Callable[[int, np.ndarray], Any] | Any = None,
    crn_metadata: dict[str, Any] | None = None,
) -> JointNullBranchResult:
    """Run CONFIG/NULL policy branches through a caller-owned continuation.

    The continuation is where a future formal collector can restore the same
    pre-action environment state and replay common environment/continuation
    randomness.  This function itself does not call an environment.  It
    restores the caller's torch RNG on exit and starts both branches from the
    same policy RNG state; NULL still consumes the dynamic-trigger draw so
    the suffix can use aligned old-policy randomness.
    """

    if not isinstance(policy, HierarchicalReconfigurationGatePolicy):
        raise TypeError("policy 必须是 HierarchicalReconfigurationGatePolicy。")
    if not callable(continuation):
        raise TypeError("continuation 必须是callable。")
    _require_torch()
    caller_rng_state = _capture_global_rng_state()
    branch_start_rng_state = _clone_state(caller_rng_state)
    samples: dict[str, HierarchicalActionSample] = {}
    returns: dict[str, GateBranchReturn] = {}
    try:
        for branch_name, forced_gate in (("CONFIG", GATE_ACTION_CONFIGURE), ("NULL", GATE_ACTION_NULL)):
            _restore_global_rng_state(_clone_state(branch_start_rng_state))
            sample = policy.sample_action(
                observation,
                deterministic=False,
                action_mask=action_mask,
                force_gate_action=forced_gate,
                consume_subpolicy_draw_for_null=True,
            )
            if int(sample.gate_action) != int(forced_gate):
                raise AssertionError("Joint-Null forced Gate action未生效。")
            branch_return = continuation(branch_name, sample)
            if not isinstance(branch_return, GateBranchReturn):
                raise TypeError("continuation 必须返回 GateBranchReturn。")
            if branch_return.branch != branch_name:
                raise ValueError("continuation 返回的branch名称不一致。")
            samples[branch_name] = sample
            returns[branch_name] = branch_return
    finally:
        _restore_global_rng_state(caller_rng_state)
    metadata = {
        "protocol": "joint_null_same_pre_action_state_crn_v1",
        "same_pre_action_state": True,
        "same_environment_randomness": True,
        "same_continuation_randomness": True,
        "policy_frozen_during_branches": True,
        "gate_forced": True,
        "null_action_strictly_all_zero": True,
        "policy_rng_branch_start_restored": True,
        "caller_rng_restored": True,
    }
    if crn_metadata:
        metadata.update(copy.deepcopy(crn_metadata))
    return JointNullBranchResult(
        config_sample=samples["CONFIG"],
        null_sample=samples["NULL"],
        config_return=returns["CONFIG"],
        null_return=returns["NULL"],
        crn_metadata=metadata,
    )


@dataclass(frozen=True)
class GateCounterfactualSample:
    """Persistable Joint-Null sample and Gate Direct-CF credit."""

    decision_id: str
    episode_index: int
    step_index: int
    slot: int
    origin_index: int
    gate_observation: np.ndarray
    gate_action: int
    p_old_gate: float
    gate_old_logprob: float
    config_joint_action: np.ndarray
    null_joint_action: np.ndarray
    return_config: float
    return_null: float
    delta_gate: float
    a_gate: float
    crn_metadata: dict[str, Any]
    collection_policy_version: int = 0
    # Formal Gate-CF decomposition.  These remain optional for the original
    # module-level fixture artifacts, while formal collection always fills all
    # three fields and validates the exact Delta identity.
    delta_q: float | None = None
    delta_cache: float | None = None
    delta_uav: float | None = None


def _validate_gate_cf_sample(sample: GateCounterfactualSample) -> None:
    if not str(sample.decision_id):
        raise ValueError("Gate CF decision_id不能为空。")
    if int(sample.episode_index) < 0 or int(sample.step_index) < 0 or int(sample.slot) < 0:
        raise ValueError("Gate CF index不能为负。")
    if int(sample.origin_index) < 0 or int(sample.collection_policy_version) < 0:
        raise ValueError("Gate CF origin/version不能为负。")
    _binary_action(sample.gate_action, "gate_action")
    p = _as_finite_float(sample.p_old_gate, "p_old_gate")
    if not 0.0 < p < 1.0:
        raise ValueError("p_old_gate 必须严格位于(0,1)。")
    old_logprob = math.log(p) if int(sample.gate_action) == 1 else math.log1p(-p)
    if not math.isclose(float(sample.gate_old_logprob), old_logprob, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("gate_old_logprob 与 p_old_gate/gate_action 不一致。")
    observation = np.asarray(sample.gate_observation, dtype=np.float32)
    if observation.ndim != 1 or not np.all(np.isfinite(observation)):
        raise ValueError("gate_observation 必须是finite一维向量。")
    config = np.asarray(sample.config_joint_action, dtype=np.int8).reshape(-1)
    null = np.asarray(sample.null_joint_action, dtype=np.int8).reshape(-1)
    if config.shape != null.shape or config.size <= 0:
        raise ValueError("Gate CF CONFIG/NULL action shape非法。")
    if not np.all(np.isin(config, [0, 1])) or not np.all(np.isin(null, [0, 1])):
        raise ValueError("Gate CF CONFIG/NULL action必须是0/1。")
    if not np.array_equal(null, np.zeros_like(null)):
        raise ValueError("Gate CF NULL joint action必须严格全零。")
    for name in ("return_config", "return_null", "delta_gate", "a_gate"):
        _as_finite_float(getattr(sample, name), name)
    expected_delta = float(sample.return_config) - float(sample.return_null)
    if not math.isclose(float(sample.delta_gate), expected_delta, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Delta_gate 必须等于 Return_CONFIG-Return_NULL。")
    expected_a = (float(sample.gate_action) - p) * expected_delta
    if not math.isclose(float(sample.a_gate), expected_a, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("A_gate 必须等于(g-p_old_gate)*Delta_gate。")
    decomposition = (sample.delta_q, sample.delta_cache, sample.delta_uav)
    if any(value is not None for value in decomposition):
        if any(value is None for value in decomposition):
            raise ValueError("Gate CF DeltaQ/DeltaCache/DeltaUAV 必须同时提供。")
        if not all(math.isfinite(float(value)) for value in decomposition):
            raise ValueError("Gate CF DeltaQ/DeltaCache/DeltaUAV 必须finite。")
        expected_decomposition = (
            float(sample.delta_q) - float(sample.delta_cache) - float(sample.delta_uav)
        )
        if not math.isclose(
            float(sample.delta_gate),
            expected_decomposition,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("Gate CF 必须满足 Delta_gate=DeltaQ-DeltaCache-DeltaUAV。")
    if not isinstance(sample.crn_metadata, dict):
        raise ValueError("crn_metadata 必须是dict。")


def build_gate_counterfactual_sample(
    *,
    decision_id: str,
    episode_index: int,
    step_index: int,
    slot: int,
    origin_index: int,
    gate_observation: Any,
    gate_action: int,
    p_old_gate: float,
    gate_old_logprob: float,
    config_joint_action: Any,
    null_joint_action: Any,
    return_config: float,
    return_null: float,
    crn_metadata: dict[str, Any],
    collection_policy_version: int = 0,
    delta_q: float | None = None,
    delta_cache: float | None = None,
    delta_uav: float | None = None,
) -> GateCounterfactualSample:
    """Construct a sample using real ``G=Q-C_cache-C_UAV`` branch returns."""

    delta_gate = float(return_config) - float(return_null)
    sample = GateCounterfactualSample(
        decision_id=str(decision_id),
        episode_index=int(episode_index),
        step_index=int(step_index),
        slot=int(slot),
        origin_index=int(origin_index),
        gate_observation=np.asarray(gate_observation, dtype=np.float32).reshape(-1).copy(),
        gate_action=_binary_action(gate_action, "gate_action"),
        p_old_gate=float(p_old_gate),
        gate_old_logprob=float(gate_old_logprob),
        config_joint_action=np.asarray(config_joint_action, dtype=np.int8).reshape(-1).copy(),
        null_joint_action=np.asarray(null_joint_action, dtype=np.int8).reshape(-1).copy(),
        return_config=float(return_config),
        return_null=float(return_null),
        delta_gate=delta_gate,
        a_gate=(float(gate_action) - float(p_old_gate)) * delta_gate,
        crn_metadata=copy.deepcopy(crn_metadata),
        collection_policy_version=int(collection_policy_version),
        delta_q=None if delta_q is None else float(delta_q),
        delta_cache=None if delta_cache is None else float(delta_cache),
        delta_uav=None if delta_uav is None else float(delta_uav),
    )
    _validate_gate_cf_sample(sample)
    return sample


def gate_counterfactual_sample_to_dict(sample: GateCounterfactualSample) -> dict[str, Any]:
    _validate_gate_cf_sample(sample)
    return {
        "decision_id": str(sample.decision_id),
        "episode_index": int(sample.episode_index),
        "step_index": int(sample.step_index),
        "slot": int(sample.slot),
        "origin_index": int(sample.origin_index),
        "gate_observation": np.asarray(sample.gate_observation, dtype=np.float32).tolist(),
        "gate_action": int(sample.gate_action),
        "p_old_gate": float(sample.p_old_gate),
        "gate_old_logprob": float(sample.gate_old_logprob),
        "config_joint_action": np.asarray(sample.config_joint_action, dtype=np.int8).tolist(),
        "null_joint_action": np.asarray(sample.null_joint_action, dtype=np.int8).tolist(),
        "Return_CONFIG": float(sample.return_config),
        "Return_NULL": float(sample.return_null),
        "Delta_gate": float(sample.delta_gate),
        "A_gate": float(sample.a_gate),
        "crn_metadata": copy.deepcopy(sample.crn_metadata),
        "collection_policy_version": int(sample.collection_policy_version),
        "DeltaQ": None if sample.delta_q is None else float(sample.delta_q),
        "DeltaCache": None if sample.delta_cache is None else float(sample.delta_cache),
        "DeltaUAV": None if sample.delta_uav is None else float(sample.delta_uav),
    }


def gate_counterfactual_sample_from_dict(value: dict[str, Any]) -> GateCounterfactualSample:
    if not isinstance(value, dict):
        raise ValueError("Gate CF artifact sample必须是dict。")
    sample = GateCounterfactualSample(
        decision_id=str(value["decision_id"]),
        episode_index=int(value["episode_index"]),
        step_index=int(value["step_index"]),
        slot=int(value["slot"]),
        origin_index=int(value["origin_index"]),
        gate_observation=np.asarray(value["gate_observation"], dtype=np.float32),
        gate_action=int(value["gate_action"]),
        p_old_gate=float(value["p_old_gate"]),
        gate_old_logprob=float(value["gate_old_logprob"]),
        config_joint_action=np.asarray(value["config_joint_action"], dtype=np.int8),
        null_joint_action=np.asarray(value["null_joint_action"], dtype=np.int8),
        return_config=float(value["Return_CONFIG"]),
        return_null=float(value["Return_NULL"]),
        delta_gate=float(value["Delta_gate"]),
        a_gate=float(value["A_gate"]),
        crn_metadata=copy.deepcopy(value["crn_metadata"]),
        collection_policy_version=int(value.get("collection_policy_version", 0)),
        delta_q=None if value.get("DeltaQ") is None else float(value["DeltaQ"]),
        delta_cache=None if value.get("DeltaCache") is None else float(value["DeltaCache"]),
        delta_uav=None if value.get("DeltaUAV") is None else float(value["DeltaUAV"]),
    )
    _validate_gate_cf_sample(sample)
    return sample


def gate_advantages(
    gate_actions: Any,
    p_old_gate: Any,
    delta_gate: Any,
) -> np.ndarray:
    """Return exactly ``(g-p_old_gate)*Delta_gate`` with no second factor."""

    actions = np.asarray(gate_actions, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(p_old_gate, dtype=np.float64).reshape(-1)
    deltas = np.asarray(delta_gate, dtype=np.float64).reshape(-1)
    if not (actions.shape == probabilities.shape == deltas.shape) or not len(actions):
        raise ValueError("gate action/probability/Delta shape必须一致且非空。")
    if not np.all(np.isin(actions, [0.0, 1.0])):
        raise ValueError("gate_actions 必须是0/1。")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("p_old_gate 必须是(0,1)内finite概率。")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("delta_gate 必须是finite。")
    return ((actions - probabilities) * deltas).astype(np.float64)


@dataclass(frozen=True)
class HierarchicalOptimizers:
    gate_optimizer: Any
    subpolicy_optimizer: Any


def build_independent_optimizers(
    policy: HierarchicalReconfigurationGatePolicy,
    learning_rate: float,
) -> HierarchicalOptimizers:
    """Build non-overlapping Adam optimizers for Gate and existing WHAT Actor."""

    _require_torch()
    lr = _as_finite_float(learning_rate, "learning_rate")
    if lr <= 0.0:
        raise ValueError("learning_rate 必须大于0。")
    gate_parameters = list(policy.gate.parameters())
    subpolicy_parameters = ppo_trigger_policy.sparse_direct_cf_actor_parameters(policy.sub_policy)
    gate_ids = {id(parameter) for parameter in gate_parameters}
    sub_ids = {id(parameter) for parameter in subpolicy_parameters}
    if gate_ids & sub_ids:
        raise AssertionError("Gate 与 WHAT Actor optimizer 参数不得重叠。")
    return HierarchicalOptimizers(
        gate_optimizer=torch.optim.Adam(gate_parameters, lr=lr),
        subpolicy_optimizer=torch.optim.Adam(subpolicy_parameters, lr=lr),
    )


def optimizer_parameter_isolation(
    policy: HierarchicalReconfigurationGatePolicy,
    optimizers: HierarchicalOptimizers,
) -> dict[str, bool]:
    gate_ids = {id(parameter) for parameter in policy.gate.parameters()}
    sub_ids = {id(parameter) for parameter in ppo_trigger_policy.sparse_direct_cf_actor_parameters(policy.sub_policy)}
    gate_optimizer_ids = {
        id(parameter)
        for group in optimizers.gate_optimizer.param_groups
        for parameter in group["params"]
    }
    sub_optimizer_ids = {
        id(parameter)
        for group in optimizers.subpolicy_optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        "gate_optimizer_only_gate": gate_optimizer_ids == gate_ids,
        "subpolicy_optimizer_only_subpolicy": sub_optimizer_ids == sub_ids,
        "optimizer_parameter_disjoint": not bool(gate_optimizer_ids & sub_optimizer_ids),
    }


@dataclass(frozen=True)
class GateUpdateResult:
    optimizer_steps: int
    policy_loss: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clip_fraction: float
    approx_kl: float
    gradient_finite: bool
    gradient_norm: float
    gate_parameter_delta_max: float
    subpolicy_parameter_delta_max: float
    p_before_mean: float
    p_after_mean: float
    p_before: tuple[float, ...]
    p_after: tuple[float, ...]
    gate_advantages: tuple[float, ...]
    entropy_bonus_applied: bool


def update_gate_on_counterfactual_batch(
    policy: HierarchicalReconfigurationGatePolicy,
    gate_optimizer: Any,
    samples: list[GateCounterfactualSample] | tuple[GateCounterfactualSample, ...],
    *,
    clip_epsilon: float = DEFAULT_GATE_CLIP_EPSILON,
    max_grad_norm: float = 0.5,
    update_epochs: int = DEFAULT_GATE_UPDATE_EPOCHS,
) -> GateUpdateResult:
    """Run the independent one-epoch Gate clipped-PPO Direct-CF update."""

    _require_torch()
    if not isinstance(policy, HierarchicalReconfigurationGatePolicy):
        raise TypeError("policy 必须是 HierarchicalReconfigurationGatePolicy。")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Gate update batch不能为空。")
    if int(update_epochs) != DEFAULT_GATE_UPDATE_EPOCHS:
        raise ValueError("Gate formal update固定为1 epoch。")
    clip = _as_finite_float(clip_epsilon, "clip_epsilon")
    max_norm = _as_finite_float(max_grad_norm, "max_grad_norm")
    if not 0.0 <= clip < 1.0 or max_norm <= 0.0:
        raise ValueError("Gate clip/max_grad_norm参数非法。")
    for sample in sample_list:
        _validate_gate_cf_sample(sample)
    decision_ids = [str(sample.decision_id) for sample in sample_list]
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("Gate update decision_id必须唯一。")
    observations = np.stack([sample.gate_observation for sample in sample_list]).astype(np.float32)
    actions = np.asarray([sample.gate_action for sample in sample_list], dtype=np.int64)
    old_probabilities = np.asarray([sample.p_old_gate for sample in sample_list], dtype=np.float32)
    old_log_probs = np.asarray([sample.gate_old_logprob for sample in sample_list], dtype=np.float32)
    deltas = np.asarray([sample.delta_gate for sample in sample_list], dtype=np.float32)
    advantages = gate_advantages(actions, old_probabilities, deltas).astype(np.float32)

    # Validate the actual Gate optimizer directly and reject any WHAT
    # parameter; the two optimizers are intentionally never shared.
    gate_ids = {id(parameter) for parameter in policy.gate.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in gate_optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != gate_ids:
        raise ValueError("gate_optimizer 必须且只能包含 Gate 参数。")

    from ppo import ppo_update

    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    for parameter in policy.sub_policy.parameters():
        parameter.grad = None
    gate_optimizer.zero_grad(set_to_none=True)
    loss = ppo_update.compute_gate_direct_cf_actor_loss(
        policy,
        observations,
        actions,
        old_log_probs,
        old_probabilities,
        deltas,
        clip,
    )
    if not bool(torch.isfinite(loss.total_loss).item()):
        raise FloatingPointError("Gate loss 非finite。")
    loss.total_loss.backward()
    subpolicy_grads = [parameter.grad for parameter in policy.sub_policy.parameters() if parameter.grad is not None]
    if subpolicy_grads:
        raise AssertionError("Gate backward 不得产生 WHAT Actor gradient。")
    gate_grads = [parameter.grad.detach() for parameter in policy.gate.parameters() if parameter.grad is not None]
    gradient_finite = bool(gate_grads) and all(bool(torch.isfinite(grad).all().item()) for grad in gate_grads)
    if not gradient_finite:
        raise FloatingPointError("Gate gradient 必须finite且非空。")
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        list(policy.gate.parameters()),
        max_norm,
    )
    gradient_norm = float(gradient_norm_tensor.detach().cpu())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("Gate gradient norm 非finite。")
    gate_optimizer.step()

    gate_delta_max = 0.0
    sub_delta_max = 0.0
    for name, parameter in policy.named_parameters():
        delta = float(torch.max(torch.abs(parameter.detach() - before[name])).cpu())
        if name.startswith("gate."):
            gate_delta_max = max(gate_delta_max, delta)
        elif name.startswith("sub_policy."):
            sub_delta_max = max(sub_delta_max, delta)
    with torch.no_grad():
        after_eval = policy.evaluate_gate_actions(
            observations,
            actions,
        )
        p_after_array = after_eval.probability_one.detach().cpu().numpy().astype(np.float64)
    p_before_array = old_probabilities.astype(np.float64)
    return GateUpdateResult(
        optimizer_steps=1,
        policy_loss=float(loss.policy_loss.detach().cpu()),
        ratio_mean=float(loss.ratio_mean),
        ratio_min=float(loss.ratio_min),
        ratio_max=float(loss.ratio_max),
        clip_fraction=float(loss.clip_fraction),
        approx_kl=float(loss.approx_kl),
        gradient_finite=bool(gradient_finite),
        gradient_norm=gradient_norm,
        gate_parameter_delta_max=float(gate_delta_max),
        subpolicy_parameter_delta_max=float(sub_delta_max),
        p_before_mean=float(np.mean(p_before_array)),
        p_after_mean=float(np.mean(p_after_array)),
        p_before=tuple(float(value) for value in p_before_array),
        p_after=tuple(float(value) for value in p_after_array),
        gate_advantages=tuple(float(value) for value in advantages),
        entropy_bonus_applied=False,
    )


def gate_update_reconstructable_fields(
    sample: GateCounterfactualSample,
) -> tuple[str, ...]:
    """Canonical field list used by the Gate causal artifact."""

    _validate_gate_cf_sample(sample)
    return (
        "gate_observation",
        "gate_action",
        "p_old_gate",
        "gate_old_logprob",
        "config_joint_action",
        "null_joint_action",
        "Return_CONFIG",
        "Return_NULL",
        "Delta_gate",
        "A_gate",
        "DeltaQ",
        "DeltaCache",
        "DeltaUAV",
        "crn_metadata",
    )
