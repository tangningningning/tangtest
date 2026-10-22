"""保存带有动作语义和下一状态合法候选的经验。"""

from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .candidate_features import ActionCandidate, CandidateOption


def _vector_tuple(value: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须为有限数值")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class PackedCandidateSet(Sequence[CandidateOption]):
    """以无损压缩保存下一状态的语义动作及候选特征。"""

    actions_zlib: bytes
    features_zlib: bytes
    candidate_count: int
    feature_dim: int

    @classmethod
    def from_candidates(
        cls, candidates: Sequence[CandidateOption]
    ) -> "PackedCandidateSet":
        options = tuple(candidates)
        if not options:
            raise ValueError("非终止转移的下一状态候选集合不能为空")
        ids = [item.action.candidate_id for item in options]
        if len(ids) != len(set(ids)):
            raise ValueError("下一状态候选集合包含重复动作 ID")
        feature_dim = len(options[0].features)
        if feature_dim <= 0 or any(len(item.features) != feature_dim for item in options):
            raise ValueError("下一状态候选特征维度不一致")
        actions = json.dumps(
            [item.action.to_dict() for item in options],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        features = np.asarray([item.features for item in options], dtype=np.float32)
        if not np.all(np.isfinite(features)):
            raise ValueError("下一状态候选特征必须为有限数值")
        return cls(
            zlib.compress(actions, level=1),
            zlib.compress(np.ascontiguousarray(features).tobytes(), level=1),
            len(options),
            feature_dim,
        )

    def __len__(self) -> int:
        return int(self.candidate_count)

    def feature_matrix(self) -> np.ndarray:
        """返回精确解压的 float32 特征矩阵。"""

        raw = zlib.decompress(self.features_zlib)
        expected_size = (
            int(self.candidate_count) * int(self.feature_dim) * np.dtype(np.float32).itemsize
        )
        if len(raw) != expected_size:
            raise ValueError("压缩的候选特征矩阵长度与形状不匹配")
        matrix = np.frombuffer(raw, dtype=np.float32).reshape(
            int(self.candidate_count), int(self.feature_dim)
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("解压的下一状态候选特征包含非有限值")
        return matrix.copy()

    def actions(self) -> tuple[ActionCandidate, ...]:
        values = json.loads(zlib.decompress(self.actions_zlib).decode("utf-8"))
        actions = tuple(ActionCandidate.from_dict(value) for value in values)
        if len(actions) != int(self.candidate_count):
            raise ValueError("压缩的候选动作数量与声明值不匹配")
        return actions

    def __iter__(self) -> Iterator[CandidateOption]:
        actions = self.actions()
        features = self.feature_matrix()
        for action, vector in zip(actions, features, strict=True):
            yield CandidateOption.create(action, vector)

    def __getitem__(self, index: int | slice) -> CandidateOption | tuple[CandidateOption, ...]:
        return tuple(iter(self))[index]


@dataclass(frozen=True)
class ReplayTransition:
    """可序列化的一个单 Slot 或多 Slot 转移。"""

    state_features: tuple[float, ...]
    action: ActionCandidate
    action_features: tuple[float, ...]
    reward: float
    next_state_features: tuple[float, ...] | None
    next_state_fingerprint: str | None
    next_legal_candidates: Sequence[CandidateOption]
    terminated: bool
    duration_slots: int

    def next_candidate_feature_matrix(self) -> np.ndarray:
        """直接恢复下一状态候选特征，避免逐候选构造 Python 对象。"""

        if isinstance(self.next_legal_candidates, PackedCandidateSet):
            return self.next_legal_candidates.feature_matrix()
        if not self.next_legal_candidates:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            [item.features for item in self.next_legal_candidates],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_features": list(self.state_features),
            "action": self.action.to_dict(),
            "action_features": list(self.action_features),
            "reward": self.reward,
            "next_state_features": None
            if self.next_state_features is None
            else list(self.next_state_features),
            "next_state_fingerprint": self.next_state_fingerprint,
            "next_legal_candidates": [item.to_dict() for item in self.next_legal_candidates],
            "terminated": self.terminated,
            "duration_slots": self.duration_slots,
        }

    def to_compact_dict(self) -> dict[str, Any]:
        """供 Checkpoint 使用的紧凑、无损经验格式。"""

        candidate_set = self.next_legal_candidates
        if self.terminated:
            packed = None
        elif isinstance(candidate_set, PackedCandidateSet):
            packed = candidate_set
        else:
            packed = PackedCandidateSet.from_candidates(candidate_set)
        return {
            "state_features": list(self.state_features),
            "action": self.action.to_dict(),
            "action_features": list(self.action_features),
            "reward": self.reward,
            "next_state_features": None
            if self.next_state_features is None
            else list(self.next_state_features),
            "next_state_fingerprint": self.next_state_fingerprint,
            "next_legal_actions_zlib": None if packed is None else packed.actions_zlib,
            "next_legal_features_zlib": None if packed is None else packed.features_zlib,
            "next_legal_candidate_count": 0
            if packed is None
            else packed.candidate_count,
            "next_legal_candidate_feature_dim": 0 if packed is None else packed.feature_dim,
            "terminated": self.terminated,
            "duration_slots": self.duration_slots,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayTransition":
        next_state = value.get("next_state_features")
        done = bool(value["terminated"])
        reward = float(value["reward"])
        duration_slots = int(value.get("duration_slots", 1))
        next_fingerprint = value.get("next_state_fingerprint")
        candidates = tuple(
            CandidateOption.from_dict(item) for item in value.get("next_legal_candidates", ())
        )
        if not np.isfinite(reward) or duration_slots <= 0:
            raise ValueError("回放奖励或持续 Slot 数无效")
        if done and (next_state is not None or next_fingerprint is not None or candidates):
            raise ValueError("终止转移不得带有下一状态候选")
        if not done and (next_state is None or not next_fingerprint or not candidates):
            raise ValueError("非终止转移必须带有对应的下一状态及合法候选")
        if not done and any(
            item.action.source_state_fingerprint != str(next_fingerprint)
            for item in candidates
        ):
            raise ValueError("回放的下一候选不属于记录的下一状态")
        return cls(
            _vector_tuple(value["state_features"], "state_features"),
            ActionCandidate.from_dict(value["action"]),
            _vector_tuple(value["action_features"], "action_features"),
            reward,
            None if next_state is None else _vector_tuple(next_state, "next_state_features"),
            None if next_fingerprint is None else str(next_fingerprint),
            candidates,
            done,
            duration_slots,
        )

    @classmethod
    def from_compact_dict(cls, value: Mapping[str, Any]) -> "ReplayTransition":
        """恢复紧凑经验并复核候选与下一状态的绑定。"""

        if "next_legal_actions_zlib" not in value:
            return cls.from_dict(value)
        next_state = value.get("next_state_features")
        done = bool(value["terminated"])
        reward = float(value["reward"])
        duration_slots = int(value["duration_slots"])
        if not np.isfinite(reward) or duration_slots <= 0:
            raise ValueError("回放奖励或持续 Slot 数无效")
        if done:
            if (
                next_state is not None
                or value.get("next_state_fingerprint") is not None
                or value.get("next_legal_actions_zlib") is not None
            ):
                raise ValueError("终止转移不得带有下一状态候选")
            candidates: Sequence[CandidateOption] = ()
        else:
            next_fingerprint = value.get("next_state_fingerprint")
            if next_state is None or not next_fingerprint:
                raise ValueError("非终止转移必须带有对应的下一状态")
            packed = PackedCandidateSet(
                bytes(value["next_legal_actions_zlib"]),
                bytes(value["next_legal_features_zlib"]),
                int(value["next_legal_candidate_count"]),
                int(value["next_legal_candidate_feature_dim"]),
            )
            if not packed:
                raise ValueError("非终止转移必须保留下一状态合法候选")
            packed.feature_matrix()
            actions = packed.actions()
            if any(
                item.source_state_fingerprint != str(next_fingerprint)
                for item in actions
            ):
                raise ValueError("回放的下一候选不属于记录的下一状态")
            if len({item.candidate_id for item in actions}) != len(actions):
                raise ValueError("回放的下一候选包含重复动作 ID")
            candidates = packed
        return cls(
            _vector_tuple(value["state_features"], "state_features"),
            ActionCandidate.from_dict(value["action"]),
            _vector_tuple(value["action_features"], "action_features"),
            reward,
            None if next_state is None else _vector_tuple(next_state, "next_state_features"),
            None if done else str(value["next_state_fingerprint"]),
            candidates,
            done,
            duration_slots,
        )


class ReplayBuffer:
    """固定容量环形回放，不保存会随候选生成顺序改变的裸编号。"""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if int(capacity) <= 0:
            raise ValueError("回放容量必须为正整数")
        self.capacity = int(capacity)
        self._items: list[ReplayTransition] = []
        self._position = 0
        self._rng = random.Random(int(seed))
        self.write_count = 0
        self.sample_calls = 0
        self.sample_count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        state_features: Any,
        action: ActionCandidate,
        action_features: Any,
        reward: float,
        next_state_features: Any | None,
        next_state_fingerprint: str | None,
        next_legal_candidates: Sequence[CandidateOption],
        terminated: bool,
        duration_slots: int = 1,
    ) -> ReplayTransition:
        if not np.isfinite(float(reward)):
            raise ValueError("经验奖励必须为有限数值")
        if int(duration_slots) <= 0:
            raise ValueError("动作持续 Slot 数必须为正整数")
        candidates = tuple(next_legal_candidates)
        if terminated:
            if next_state_features is not None or next_state_fingerprint is not None or candidates:
                raise ValueError("终止转移不得提供下一状态或下一候选")
            next_state = None
        else:
            if next_state_features is None or not next_state_fingerprint or not candidates:
                raise ValueError("非终止转移必须提供与下一状态匹配的合法候选")
            next_state = _vector_tuple(next_state_features, "next_state_features")
            if any(not np.all(np.isfinite(np.asarray(item.features))) for item in candidates):
                raise ValueError("下一状态候选特征必须为有限数值")
            if any(
                item.action.source_state_fingerprint != str(next_state_fingerprint)
                for item in candidates
            ):
                raise ValueError("下一状态候选与经验中的下一状态指纹不一致")
        if action.source_state_fingerprint == "":
            raise ValueError("回放动作缺少其来源状态指纹")
        transition = ReplayTransition(
            _vector_tuple(state_features, "state_features"),
            ActionCandidate.from_dict(action.to_dict()),
            _vector_tuple(action_features, "action_features"),
            float(reward),
            next_state,
            None if terminated else str(next_state_fingerprint),
            () if terminated else PackedCandidateSet.from_candidates(candidates),
            bool(terminated),
            int(duration_slots),
        )
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity
        self.write_count += 1
        return transition

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if int(batch_size) <= 0 or int(batch_size) > len(self._items):
            raise ValueError("批量大小必须在当前回放样本数范围内")
        sampled = self._rng.sample(self._items, int(batch_size))
        self.sample_calls += 1
        self.sample_count += int(batch_size)
        return sampled

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "position": self._position,
            "items": [item.to_compact_dict() for item in self._items],
            "rng_state": self._rng.getstate(),
            "write_count": self.write_count,
            "sample_calls": self.sample_calls,
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        capacity = int(state["capacity"])
        items = [ReplayTransition.from_compact_dict(item) for item in state["items"]]
        if capacity != self.capacity or len(items) > self.capacity:
            raise ValueError("回放状态容量与当前缓冲区不匹配")
        self._items = items
        self._position = int(state["position"]) % self.capacity
        self._rng.setstate(state["rng_state"])
        self.write_count = int(state.get("write_count", len(items)))
        self.sample_calls = int(state.get("sample_calls", 0))
        self.sample_count = int(state.get("sample_count", 0))


__all__ = ["ReplayTransition", "ReplayBuffer"]
