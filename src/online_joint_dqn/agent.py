"""面向动态合法候选集合的集中式 Double DQN。"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_features import CandidateOption
from .replay_buffer import ReplayTransition


class CandidateQNetwork(nn.Module):
    """输入状态与单个候选动作，输出该候选的一个 Q 值。"""

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
        for before, after in zip(widths[:-1], widths[1:]):
            layers.extend((nn.Linear(before, after), nn.ReLU()))
        layers.append(nn.Linear(widths[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        if state_features.ndim != 2 or candidate_features.ndim != 2:
            raise ValueError("Q 网络输入必须为二维 batch 张量")
        if state_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("状态与候选特征 batch 大小必须一致")
        return self.network(torch.cat((state_features, candidate_features), dim=1)).squeeze(-1)


@dataclass(frozen=True)
class ActionSelection:
    candidate: CandidateOption
    q_value: float | None
    explored: bool
    epsilon: float
    candidate_counts_by_operation: Mapping[str, int]
    q_statistics_by_operation: Mapping[str, Mapping[str, Any]]


class DoubleDQNAgent:
    """复用 DQN 在线/目标网训练骨架并实现候选条件化 Double DQN。"""

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        *,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
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
        self.state_dim = int(state_dim)
        self.candidate_dim = int(candidate_dim)
        self.gamma = float(gamma)
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
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=float(learning_rate))
        self.gradient_steps = 0

    def score_candidates(
        self,
        state_features: Any,
        candidates: Sequence[CandidateOption],
        *,
        network: nn.Module | None = None,
    ) -> np.ndarray:
        """只为调用方提供的当前合法候选打分。"""

        if not candidates:
            raise ValueError("合法候选集合不能为空，KEEP_STAY 应始终存在")
        state = np.asarray(state_features, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError("状态特征维度不匹配")
        actions = np.asarray([item.features for item in candidates], dtype=np.float32)
        if actions.shape != (len(candidates), self.candidate_dim):
            raise ValueError("候选特征维度不匹配")
        selected_network = self.online_network if network is None else network
        selected_network.eval()
        with torch.no_grad():
            states = torch.as_tensor(
                np.repeat(state[None, :], len(candidates), axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            scores = selected_network(states, action_tensor).detach().cpu().numpy()
        if scores.shape != (len(candidates),) or not np.all(np.isfinite(scores)):
            raise FloatingPointError("Q 网络对合法候选返回了无效分数")
        return scores.astype(np.float64, copy=False)

    def select_action(
        self,
        state_features: Any,
        legal_candidates: Sequence[CandidateOption],
        *,
        epsilon: float = 0.0,
        explore: bool = False,
    ) -> ActionSelection:
        """探索时先均匀抽取动作类别，贪心时比较全部合法候选。"""

        if not legal_candidates:
            raise ValueError("动作选择要求非空合法候选集合")
        operation_order = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
        candidates_by_operation: dict[str, list[CandidateOption]] = {
            operation: [] for operation in operation_order
        }
        for candidate in legal_candidates:
            operation = candidate.action.operation
            if operation not in candidates_by_operation:
                raise ValueError(f"候选包含未知操作类别：{operation}")
            candidates_by_operation[operation].append(candidate)
        if not candidates_by_operation["KEEP_STAY"]:
            raise ValueError("合法候选集合必须始终包含 KEEP_STAY")

        chance = float(epsilon) if explore else 0.0
        if not 0.0 <= chance <= 1.0:
            raise ValueError("epsilon 必须位于 [0, 1]")
        # 探索时也计算候选 Q 统计，确保每一步可解释；该评分不执行系统求解。
        scores = self.score_candidates(state_features, legal_candidates)
        candidate_counts = {
            operation: len(candidates_by_operation[operation])
            for operation in operation_order
        }
        q_statistics: dict[str, dict[str, Any]] = {}
        for operation in operation_order:
            indices = [
                index
                for index, candidate in enumerate(legal_candidates)
                if candidate.action.operation == operation
            ]
            if not indices:
                q_statistics[operation] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "max_candidate_id": None,
                }
                continue
            operation_scores = scores[indices]
            best_offset = int(np.argmax(operation_scores))
            q_statistics[operation] = {
                "count": len(indices),
                "mean": float(np.mean(operation_scores)),
                "std": float(np.std(operation_scores)),
                "min": float(np.min(operation_scores)),
                "max": float(operation_scores[best_offset]),
                "max_candidate_id": legal_candidates[
                    indices[best_offset]
                ].action.candidate_id,
            }
        if explore and float(self.rng.random()) < chance:
            available_operations = tuple(
                operation
                for operation in operation_order
                if candidates_by_operation[operation]
            )
            operation_index = int(self.rng.integers(0, len(available_operations)))
            selected_operation = available_operations[operation_index]
            operation_candidates = candidates_by_operation[selected_operation]
            candidate_index = int(self.rng.integers(0, len(operation_candidates)))
            candidate = operation_candidates[candidate_index]
            score_index = next(
                index
                for index, option in enumerate(legal_candidates)
                if option.action.candidate_id == candidate.action.candidate_id
            )
            return ActionSelection(
                candidate,
                float(scores[score_index]),
                True,
                chance,
                candidate_counts,
                q_statistics,
            )
        index = int(np.argmax(scores))
        return ActionSelection(
            legal_candidates[index],
            float(scores[index]),
            False,
            chance,
            candidate_counts,
            q_statistics,
        )

    def compute_td_target(self, transition: ReplayTransition) -> torch.Tensor:
        """按 Double DQN 规则计算一个转移的 TD 目标。"""

        reward = torch.tensor(float(transition.reward), dtype=torch.float32, device=self.device)
        if transition.terminated:
            return reward
        if transition.next_state_features is None or not transition.next_legal_candidates:
            raise ValueError("非终止转移缺少匹配的下一状态候选")
        state = torch.as_tensor(
            np.asarray(transition.next_state_features, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        candidate_features = transition.next_candidate_feature_matrix()
        candidates = torch.as_tensor(
            candidate_features,
            dtype=torch.float32,
            device=self.device,
        )
        states = state.expand(len(transition.next_legal_candidates), -1)
        with torch.no_grad():
            online_values = self.online_network(states, candidates)
            selected_index = int(torch.argmax(online_values).item())
            target_value = self.target_network(
                state,
                candidates[selected_index : selected_index + 1],
            ).squeeze(0)
            discount = self.gamma ** int(transition.duration_slots)
            return reward + float(discount) * target_value

    def update(self, transitions: Sequence[ReplayTransition]) -> dict[str, float]:
        """执行一次 Huber TD 更新、梯度裁剪和目标网同步。"""

        if not transitions:
            raise ValueError("更新批次不能为空")
        states = np.asarray([item.state_features for item in transitions], dtype=np.float32)
        actions = np.asarray([item.action_features for item in transitions], dtype=np.float32)
        if states.shape != (len(transitions), self.state_dim):
            raise ValueError("经验状态特征维度不匹配")
        if actions.shape != (len(transitions), self.candidate_dim):
            raise ValueError("经验动作特征维度不匹配")
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        targets = torch.stack([self.compute_td_target(item) for item in transitions]).detach()
        old_values = self.online_network(state_tensor, action_tensor)
        loss = F.smooth_l1_loss(old_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.gradient_steps += 1
        synced = self.gradient_steps % self.target_network_frequency == 0
        if synced:
            self._update_target_network()
        return {
            "td_loss": float(loss.detach().cpu()),
            "q_value_mean": float(old_values.detach().mean().cpu()),
            "td_target_mean": float(targets.mean().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "gradient_step": float(self.gradient_steps),
            "target_network_synced": float(synced),
        }

    def _update_target_network(self) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_network.parameters(), self.online_network.parameters(), strict=True
            ):
                target_parameter.lerp_(online_parameter, self.target_tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "candidate_dim": self.candidate_dim,
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
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["state_dim"]) != self.state_dim or int(state["candidate_dim"]) != self.candidate_dim:
            raise ValueError("模型状态维度与当前候选条件化网络不一致")
        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
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
    "ActionSelection",
    "DoubleDQNAgent",
    "linear_epsilon_schedule",
]
