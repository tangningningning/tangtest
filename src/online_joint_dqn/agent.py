"""合法候选均值中心化的互斥动作 Dueling Double DQN。"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_features import (
    CANDIDATE_FEATURE_CONTRACT_VERSION,
    NETWORK_ARCHITECTURE_VERSION,
    OPERATIONS,
    STATE_FEATURE_CONTRACT_VERSION,
    CandidateOption,
)
from .replay_buffer import ReplayTransition


class CandidateQNetwork(nn.Module):
    """输入状态与类别内候选特征，输出候选优势分数。"""

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        if int(state_dim) <= 0 or int(candidate_dim) <= 0:
            raise ValueError("状态和候选特征维度必须为正数")
        widths = [int(state_dim) + int(candidate_dim), *(int(value) for value in hidden_sizes)]
        if any(value <= 0 for value in widths):
            raise ValueError("网络宽度必须为正数")
        layers: list[nn.Module] = []
        for before, after in zip(widths[:-1], widths[1:], strict=True):
            layers.extend((nn.Linear(before, after), nn.ReLU()))
        layers.append(nn.Linear(widths[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        if state_features.ndim != 2 or candidate_features.ndim != 2:
            raise ValueError("候选评分网络输入必须为二维 batch 张量")
        if state_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("状态与候选特征 batch 大小必须一致")
        return self.network(
            torch.cat((state_features, candidate_features), dim=1)
        ).squeeze(-1)


class ActionCategoryQNetwork(nn.Module):
    """只根据状态估计 KEEP、缓存更新和 UAV 移动的类别价值。"""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        if int(state_dim) <= 0:
            raise ValueError("状态特征维度必须为正数")
        widths = [int(state_dim), *(int(value) for value in hidden_sizes)]
        if any(value <= 0 for value in widths):
            raise ValueError("网络宽度必须为正数")
        layers: list[nn.Module] = []
        for before, after in zip(widths[:-1], widths[1:], strict=True):
            layers.extend((nn.Linear(before, after), nn.ReLU()))
        layers.append(nn.Linear(widths[-1], len(OPERATIONS)))
        self.network = nn.Sequential(*layers)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        if state_features.ndim != 2:
            raise ValueError("动作类别网络输入必须为二维 batch 张量")
        return self.network(state_features)


@dataclass(frozen=True)
class ActionSelection:
    candidate: CandidateOption
    q_value: float
    category_q_value: float
    candidate_q_value: float
    explored: bool
    epsilon: float
    candidate_counts_by_operation: Mapping[str, int]
    q_statistics_by_operation: Mapping[str, Mapping[str, Any]]
    q_values_by_action_category: Mapping[str, float | None]
    candidate_advantage: float


CandidateRebuilder = Callable[[Mapping[str, Any]], Sequence[CandidateOption]]


class DoubleDQNAgent:
    """用完整合法候选 Q 排序，并以 Double DQN 更新类别价值和候选优势。"""

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        *,
        learning_rate: float = 2.5e-4,
        gamma: float = 1.0,
        target_network_frequency: int = 500,
        target_tau: float = 1.0,
        max_grad_norm: float = 10.0,
        hidden_sizes: Sequence[int] = (256, 128),
        seed: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma 必须位于 [0, 1]")
        if int(target_network_frequency) <= 0 or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("目标网络同步参数非法")
        if float(max_grad_norm) <= 0.0:
            raise ValueError("梯度裁剪上限必须为正数")
        if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate 必须为正的有限数值")
        self.state_dim = int(state_dim)
        self.candidate_dim = int(candidate_dim)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.target_network_frequency = int(target_network_frequency)
        self.target_tau = float(target_tau)
        self.max_grad_norm = float(max_grad_norm)
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        self.rng = np.random.default_rng(self.seed)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online_network = CandidateQNetwork(
            self.state_dim, self.candidate_dim, hidden_sizes
        ).to(self.device)
        self.target_network = copy.deepcopy(self.online_network).to(self.device)
        self.target_network.eval()
        self.online_category_network = ActionCategoryQNetwork(
            self.state_dim, hidden_sizes
        ).to(self.device)
        self.target_category_network = copy.deepcopy(
            self.online_category_network
        ).to(self.device)
        self.target_category_network.eval()
        self.optimizer = torch.optim.Adam(
            [
                *self.online_network.parameters(),
                *self.online_category_network.parameters(),
            ],
            lr=self.learning_rate,
        )
        self.gradient_steps = 0

    def score_candidates(
        self,
        state_features: Any,
        candidates: Sequence[CandidateOption],
        *,
        network: nn.Module | None = None,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """分块评分，避免完整枚举候选时构造巨型重复状态张量。"""

        if not candidates:
            raise ValueError("类别内合法候选集合不能为空")
        state = np.asarray(state_features, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError("状态特征维度不匹配")
        if int(chunk_size) <= 0:
            raise ValueError("候选评分 chunk_size 必须为正数")
        selected_network = self.online_network if network is None else network
        was_training = selected_network.training
        selected_network.eval()
        all_scores = np.empty(len(candidates), dtype=np.float64)
        try:
            with torch.no_grad():
                for start in range(0, len(candidates), int(chunk_size)):
                    end = min(start + int(chunk_size), len(candidates))
                    actions = np.asarray(
                        [item.features for item in candidates[start:end]],
                        dtype=np.float32,
                    )
                    if actions.shape != (end - start, self.candidate_dim):
                        raise ValueError("候选特征维度不匹配")
                    states = np.broadcast_to(
                        state, (end - start, self.state_dim)
                    ).copy()
                    values = selected_network(
                        torch.as_tensor(states, dtype=torch.float32, device=self.device),
                        torch.as_tensor(actions, dtype=torch.float32, device=self.device),
                    ).detach().cpu().numpy()
                    all_scores[start:end] = values
        finally:
            selected_network.train(was_training)
        if not np.all(np.isfinite(all_scores)):
            raise FloatingPointError("候选评分网络返回了非有限值")
        return all_scores

    def score_action_categories(
        self,
        state_features: Any,
        *,
        network: nn.Module | None = None,
    ) -> dict[str, float]:
        state = np.asarray(state_features, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError("动作类别评分状态维度或数值无效")
        selected_network = (
            self.online_category_network if network is None else network
        )
        was_training = selected_network.training
        selected_network.eval()
        try:
            with torch.no_grad():
                tensor = torch.as_tensor(
                    state[None, :], dtype=torch.float32, device=self.device
                )
                values = selected_network(tensor).squeeze(0).cpu().numpy()
        finally:
            selected_network.train(was_training)
        if values.shape != (len(OPERATIONS),) or not np.all(np.isfinite(values)):
            raise FloatingPointError("类别价值头返回了非有限值或错误维度")
        return {operation: float(values[index]) for index, operation in enumerate(OPERATIONS)}

    def score_q_values(
        self,
        state_features: Any,
        legal_candidates: Sequence[CandidateOption],
        *,
        candidate_network: nn.Module | None = None,
        category_network: nn.Module | None = None,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        """用同一合法候选集合计算类别均值中心化 Q、类别均值和优势。"""

        if not legal_candidates:
            raise ValueError("Q 评分要求非空合法候选集合")
        raw_advantages = self.score_candidates(
            state_features, legal_candidates, network=candidate_network
        )
        category_values = self.score_action_categories(
            state_features, network=category_network
        )
        q_values = np.empty_like(raw_advantages)
        centered_advantages = np.empty_like(raw_advantages)
        indices_by_operation: dict[str, list[int]] = {
            operation: [] for operation in OPERATIONS
        }
        for index, candidate in enumerate(legal_candidates):
            operation = candidate.action.operation
            if operation not in indices_by_operation:
                raise ValueError(f"候选包含未知操作类别：{operation}")
            indices_by_operation[operation].append(index)
        for operation, indices in indices_by_operation.items():
            if not indices:
                continue
            values = raw_advantages[indices]
            centered = values - float(np.mean(values))
            centered_advantages[indices] = centered
            q_values[indices] = category_values[operation] + centered
        if not np.all(np.isfinite(q_values)):
            raise FloatingPointError("候选 Q 值包含非有限数值")
        return q_values, category_values, centered_advantages

    def select_action(
        self,
        state_features: Any,
        legal_candidates: Sequence[CandidateOption],
        *,
        epsilon: float = 0.0,
        explore: bool = False,
    ) -> ActionSelection:
        """贪心比较全部合法候选；探索时先均衡选类别再选候选。"""

        if not legal_candidates:
            raise ValueError("动作选择要求非空合法候选集合")
        candidates_by_operation: dict[str, list[tuple[int, CandidateOption]]] = {
            operation: [] for operation in OPERATIONS
        }
        for index, candidate in enumerate(legal_candidates):
            operation = candidate.action.operation
            if operation not in candidates_by_operation:
                raise ValueError(f"候选包含未知操作类别：{operation}")
            candidates_by_operation[operation].append((index, candidate))
        if not candidates_by_operation["KEEP_STAY"]:
            raise ValueError("合法候选集合必须始终包含 KEEP_STAY")

        chance = float(epsilon) if explore else 0.0
        if not 0.0 <= chance <= 1.0:
            raise ValueError("epsilon 必须位于 [0, 1]")
        available_operations = tuple(
            operation for operation in OPERATIONS if candidates_by_operation[operation]
        )
        action_q_values, category_values, centered_advantages = self.score_q_values(
            state_features, legal_candidates
        )
        candidate_counts = {
            operation: len(candidates_by_operation[operation])
            for operation in OPERATIONS
        }
        q_statistics: dict[str, dict[str, Any]] = {}
        for operation in OPERATIONS:
            indexed_options = candidates_by_operation[operation]
            if not indexed_options:
                q_statistics[operation] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "max_candidate_id": None,
                }
                continue
            indices = [index for index, _ in indexed_options]
            q_values = action_q_values[indices]
            q_statistics[operation] = {
                "count": len(indices),
                "mean": float(np.mean(q_values)),
                "std": float(np.std(q_values)),
                "min": float(np.min(q_values)),
                "max": float(np.max(q_values)),
                "max_candidate_id": indexed_options[int(np.argmax(q_values))][1].action.candidate_id,
            }

        explored = bool(explore and float(self.rng.random()) < chance)
        if explored:
            selected_operation = available_operations[
                int(self.rng.integers(0, len(available_operations)))
            ]
            operation_options = candidates_by_operation[selected_operation]
            selected_index, candidate = operation_options[
                int(self.rng.integers(0, len(operation_options)))
            ]
        else:
            selected_index = int(np.argmax(action_q_values))
            candidate = legal_candidates[selected_index]
            selected_operation = candidate.action.operation
            operation_options = candidates_by_operation[selected_operation]
        category_q = category_values[selected_operation]
        candidate_q = float(action_q_values[selected_index])
        return ActionSelection(
            candidate=candidate,
            q_value=candidate_q,
            category_q_value=category_q,
            candidate_q_value=candidate_q,
            explored=explored,
            epsilon=chance,
            candidate_counts_by_operation=candidate_counts,
            q_statistics_by_operation=q_statistics,
            q_values_by_action_category={
                operation: (
                    category_values[operation]
                    if candidates_by_operation[operation]
                    else None
                )
                for operation in OPERATIONS
            },
            candidate_advantage=float(centered_advantages[selected_index]),
        )

    def _mean_advantage_tensor(
        self, state: torch.Tensor, candidates: Sequence[CandidateOption],
        *, chunk_size: int = 2048,
    ) -> torch.Tensor:
        """计算同类别合法候选优势均值，并保留完整反向梯度。"""

        sums: list[torch.Tensor] = []
        value_count = 0
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size]
            features = torch.as_tensor(
                np.asarray([item.features for item in chunk], dtype=np.float32),
                device=self.device,
            )
            values = self.online_network(state[None, :].expand(len(chunk), -1), features)
            sums.append(values.sum())
            value_count += len(chunk)
        if value_count == 0:
            raise ValueError("所选动作类别缺少合法候选")
        return torch.stack(sums).sum() / float(value_count)

    def _next_action_target(
        self,
        transition: ReplayTransition,
        candidate_rebuilder: CandidateRebuilder | None,
    ) -> tuple[torch.Tensor, str, str]:
        if not np.isfinite(transition.reward) or int(transition.duration_slots) <= 0:
            raise ValueError("TD 转移奖励或持续时长非法")
        if transition.terminated:
            return (
                torch.tensor(
                    float(transition.reward), dtype=torch.float32, device=self.device
                ),
                "TERMINAL",
                "",
            )
        if transition.next_state_features is None:
            raise ValueError("非终止转移缺少下一状态")
        candidates = transition.materialize_next_candidates(candidate_rebuilder)
        if not candidates:
            raise ValueError("非终止转移缺少重建后的合法下一状态候选")
        candidates_by_operation: dict[str, list[CandidateOption]] = {
            operation: [] for operation in OPERATIONS
        }
        for candidate in candidates:
            if candidate.action.operation not in candidates_by_operation:
                raise ValueError("回放重建出未知的下一动作类别")
            candidates_by_operation[candidate.action.operation].append(candidate)
        if not candidates_by_operation["KEEP_STAY"]:
            raise ValueError("重建的下一状态候选缺少 KEEP_STAY")
        state = np.asarray(transition.next_state_features, dtype=np.float32)
        q_values, _, _ = self.score_q_values(state, candidates)
        selected_index = int(np.argmax(q_values))
        selected_candidate = candidates[selected_index]
        selected_operation = selected_candidate.action.operation
        category_candidates = candidates_by_operation[selected_operation]
        state_tensor = torch.as_tensor(
            state[None, :], dtype=torch.float32, device=self.device
        )
        category_index = OPERATIONS.index(selected_operation)
        with torch.no_grad():
            target_advantages = self.score_candidates(
                state, category_candidates, network=self.target_network
            )
            selected_category_index = next(
                index
                for index, candidate in enumerate(category_candidates)
                if candidate.action.candidate_id == selected_candidate.action.candidate_id
            )
            target_centered_advantage = float(
                target_advantages[selected_category_index]
                - np.mean(target_advantages)
            )
            target_category = self.target_category_network(state_tensor)[0, category_index]
            discount = self.gamma ** int(transition.duration_slots)
            target = torch.tensor(
                float(transition.reward), dtype=torch.float32, device=self.device
            ) + float(discount) * (target_category + target_centered_advantage)
        if not torch.isfinite(target):
            raise FloatingPointError("Double DQN 两阶段 TD 目标包含非有限值")
        return target, selected_operation, selected_candidate.action.candidate_id

    def compute_td_target(
        self,
        transition: ReplayTransition,
        *,
        candidate_rebuilder: CandidateRebuilder | None = None,
    ) -> torch.Tensor:
        """在线类别头与候选头顺序选动作，目标网络估计该动作的价值。"""

        return self._next_action_target(transition, candidate_rebuilder)[0]

    def update(
        self,
        transitions: Sequence[ReplayTransition],
        *,
        candidate_rebuilder: CandidateRebuilder | None = None,
    ) -> dict[str, Any]:
        if not transitions:
            raise ValueError("更新批次不能为空")
        states = np.asarray([item.state_features for item in transitions], dtype=np.float32)
        actions = np.asarray([item.action_features for item in transitions], dtype=np.float32)
        if states.shape != (len(transitions), self.state_dim):
            raise ValueError("经验状态特征维度不匹配")
        if actions.shape != (len(transitions), self.candidate_dim):
            raise ValueError("经验动作特征维度不匹配")
        action_categories = np.asarray(
            [OPERATIONS.index(item.action.operation) for item in transitions],
            dtype=np.int64,
        )
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        category_tensor = torch.as_tensor(
            action_categories, dtype=torch.long, device=self.device
        )
        target_details = [
            self._next_action_target(item, candidate_rebuilder)
            for item in transitions
        ]
        targets = torch.stack([item[0] for item in target_details]).detach()
        category_matrix = self.online_category_network(state_tensor)
        chosen_category_values = category_matrix.gather(
            1, category_tensor[:, None]
        ).squeeze(1)
        chosen_candidate_values = self.online_network(state_tensor, action_tensor)
        class_means: list[torch.Tensor] = []
        for index, transition in enumerate(transitions):
            legal_candidates = transition.materialize_current_candidates(candidate_rebuilder)
            class_candidates = [
                item for item in legal_candidates
                if item.action.operation == transition.action.operation
            ]
            recorded = next(
                (item for item in class_candidates
                 if item.action.candidate_id == transition.action.candidate_id),
                None,
            )
            if recorded is None or recorded.action.to_dict() != transition.action.to_dict() or not np.array_equal(
                np.asarray(recorded.features, dtype=np.float32), actions[index]
            ):
                raise ValueError("当前回放动作/特征与重建的合法候选不一致")
            class_means.append(self._mean_advantage_tensor(state_tensor[index], class_candidates))
        centered_advantages = chosen_candidate_values - torch.stack(class_means)
        combined_values = chosen_category_values + centered_advantages
        # 按类别均值中心化后，V 表示合法候选平均 Q；贪心必须比较完整 Q。
        loss = F.smooth_l1_loss(combined_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Dueling Double DQN TD loss 非有限")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                *self.online_network.parameters(),
                *self.online_category_network.parameters(),
            ],
            self.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Dueling Double DQN 梯度非有限，未执行优化器更新")
        self.optimizer.step()
        self.gradient_steps += 1
        synced = self.gradient_steps % self.target_network_frequency == 0
        if synced:
            self._update_target_network()
        return {
            "td_loss": float(loss.detach().cpu()),
            "category_td_loss": None,
            "candidate_advantage_loss": None,
            "td_loss_semantics": "joint_legal_candidate_mean_centered_dueling_q_huber",
            "q_value_mean": float(combined_values.detach().mean().cpu()),
            "category_q_value_mean": float(chosen_category_values.detach().mean().cpu()),
            "candidate_q_value_mean": float(combined_values.detach().mean().cpu()),
            "candidate_advantage_mean": float(centered_advantages.detach().mean().cpu()),
            "td_target_mean": float(targets.mean().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "gradient_step": float(self.gradient_steps),
            "target_network_synced": float(synced),
            "bootstrap_category_selection": [item[1] for item in target_details],
            "bootstrap_candidate_selection": [item[2] for item in target_details],
        }

    def _update_target_network(self) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_network.parameters(),
                self.online_network.parameters(),
                strict=True,
            ):
                target_parameter.lerp_(online_parameter, self.target_tau)
            for target_parameter, online_parameter in zip(
                self.target_category_network.parameters(),
                self.online_category_network.parameters(),
                strict=True,
            ):
                target_parameter.lerp_(online_parameter, self.target_tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_state_schema_version": 4,
            "q_aggregation_contract": "legal_candidate_mean_centered_dueling_v1",
            "state_feature_contract_version": STATE_FEATURE_CONTRACT_VERSION,
            "candidate_feature_contract_version": CANDIDATE_FEATURE_CONTRACT_VERSION,
            "network_architecture_version": NETWORK_ARCHITECTURE_VERSION,
            "state_dim": self.state_dim,
            "candidate_dim": self.candidate_dim,
            "learning_rate": self.learning_rate,
            "optimizer_type": type(self.optimizer).__name__,
            "gamma": self.gamma,
            "target_network_frequency": self.target_network_frequency,
            "target_tau": self.target_tau,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "gradient_steps": self.gradient_steps,
            "agent_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "online_category_network": self.online_category_network.state_dict(),
            "target_category_network": self.target_category_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("agent_state_schema_version", -1)) != 4:
            raise ValueError("旧版 DQN Checkpoint 不符合合法候选均值中心化合同，拒绝恢复")
        if state.get("q_aggregation_contract") != "legal_candidate_mean_centered_dueling_v1":
            raise ValueError("Checkpoint Q 聚合语义不兼容，拒绝恢复")
        expected_contract = {
            "state_feature_contract_version": STATE_FEATURE_CONTRACT_VERSION,
            "candidate_feature_contract_version": CANDIDATE_FEATURE_CONTRACT_VERSION,
            "network_architecture_version": NETWORK_ARCHITECTURE_VERSION,
        }
        missing = [key for key in expected_contract if key not in state]
        if missing:
            raise ValueError(
                "Checkpoint 缺少 DQN 特征/网络合同版本，属于旧版输入合同，拒绝恢复："
                + ", ".join(missing)
            )
        mismatched = [
            f"{key}={state.get(key)!r}（期望 {expected!r}）"
            for key, expected in expected_contract.items()
            if state.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "Checkpoint 的 DQN 特征/网络合同与当前输入不兼容，拒绝恢复："
                + "; ".join(mismatched)
            )
        if (
            int(state["state_dim"]) != self.state_dim
            or int(state["candidate_dim"]) != self.candidate_dim
        ):
            raise ValueError("模型状态维度与当前两阶段网络不一致，拒绝恢复")
        if float(state.get("learning_rate", float("nan"))) != self.learning_rate:
            raise ValueError("Checkpoint 学习率与当前实验不一致，拒绝恢复")
        if state.get("optimizer_type") != type(self.optimizer).__name__:
            raise ValueError("Checkpoint 优化器类型与当前实验不一致，拒绝恢复")
        for name in ("gamma", "target_network_frequency", "target_tau", "max_grad_norm"):
            if state.get(name) != getattr(self, name):
                raise ValueError(f"Checkpoint {name} 与当前学习合同不一致，拒绝恢复")
        required = (
            "online_category_network",
            "target_category_network",
        )
        if any(name not in state for name in required):
            raise ValueError("Checkpoint 缺少动作类别网络参数，拒绝恢复")
        self.online_network.load_state_dict(state["online_network"], strict=True)
        self.target_network.load_state_dict(state["target_network"], strict=True)
        self.online_category_network.load_state_dict(
            state["online_category_network"], strict=True
        )
        self.target_category_network.load_state_dict(
            state["target_category_network"], strict=True
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.gradient_steps = int(state["gradient_steps"])
        if "agent_rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["agent_rng_state"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and state.get("torch_cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def linear_epsilon_schedule(
    start: float,
    end: float,
    duration: int,
    timestep: int,
) -> float:
    """返回受控线性 epsilon-greedy 探索率。"""

    if not 0.0 <= float(end) <= float(start) <= 1.0:
        raise ValueError("epsilon 端点必须满足 0 <= end <= start <= 1")
    if int(duration) <= 0:
        raise ValueError("epsilon 衰减步数必须为正数")
    fraction = min(max(int(timestep), 0) / float(duration), 1.0)
    return float(start) + fraction * (float(end) - float(start))


__all__ = [
    "CandidateQNetwork",
    "ActionCategoryQNetwork",
    "ActionSelection",
    "DoubleDQNAgent",
    "linear_epsilon_schedule",
]
