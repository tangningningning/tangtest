"""保存转移及可确定性重建的下一状态候选上下文。"""

from __future__ import annotations

import hashlib
import pickle
import random
import zlib
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .candidate_features import ActionCandidate, CandidateOption


def _vector_tuple(value: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须为有限数值")
    return tuple(float(item) for item in array)


def _candidate_ids_sha256(candidates: Sequence[CandidateOption]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.action.candidate_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


CandidateRebuilder = Callable[[Mapping[str, Any]], Sequence[CandidateOption]]


@dataclass(frozen=True)
class ReplayTransition:
    """经验主体只保留所选动作和下一状态重建上下文，不保留候选特征矩阵。"""

    state_features: tuple[float, ...]
    action: ActionCandidate
    action_features: tuple[float, ...]
    reward: float
    next_state_features: tuple[float, ...] | None
    next_state_fingerprint: str | None
    next_legal_candidates: Sequence[CandidateOption]
    terminated: bool
    duration_slots: int
    next_candidate_rebuild_context_zlib: bytes | None = None
    next_candidate_count: int = 0
    next_candidate_ids_sha256: str | None = None
    current_legal_candidates: Sequence[CandidateOption] = ()
    current_candidate_rebuild_context_zlib: bytes | None = None
    current_candidate_count: int = 0
    current_candidate_ids_sha256: str | None = None

    def materialize_current_candidates(
        self,
        candidate_rebuilder: CandidateRebuilder | None,
    ) -> tuple[CandidateOption, ...]:
        """用当前状态合同重建候选，供类别内优势归一化使用。"""

        current_view = replace(
            self,
            next_state_features=self.state_features,
            next_state_fingerprint=self.action.source_state_fingerprint,
            next_legal_candidates=self.current_legal_candidates,
            terminated=False,
            next_candidate_rebuild_context_zlib=self.current_candidate_rebuild_context_zlib,
            next_candidate_count=self.current_candidate_count,
            next_candidate_ids_sha256=self.current_candidate_ids_sha256,
        )
        return current_view.materialize_next_candidates(candidate_rebuilder)

    def materialize_next_candidates(
        self,
        candidate_rebuilder: CandidateRebuilder | None,
    ) -> tuple[CandidateOption, ...]:
        """从预测上下文重建候选并校验动作集合身份。"""

        if self.terminated:
            return ()
        if self.next_legal_candidates:
            candidates = tuple(self.next_legal_candidates)
        else:
            if (
                candidate_rebuilder is None
                or self.next_candidate_rebuild_context_zlib is None
            ):
                raise ValueError("回放转移缺少下一候选重建器或合法上下文")
            context = pickle.loads(
                zlib.decompress(self.next_candidate_rebuild_context_zlib)
            )
            if not isinstance(context, Mapping):
                raise ValueError("回放候选重建上下文必须是映射")
            if context.get("source_state_fingerprint") != str(
                self.next_state_fingerprint
            ):
                raise ValueError("回放候选上下文与下一状态指纹不匹配")
            if "state_features" in context:
                raise ValueError("紧凑回放上下文不得重复保存下一状态特征")
            if self.next_state_features is None:
                raise ValueError("候选重建缺少已保存的下一状态特征")
            context = dict(context)
            context["state_features"] = np.asarray(
                self.next_state_features, dtype=np.float32
            )
            candidates = tuple(candidate_rebuilder(context))
        if not candidates:
            raise ValueError("重建的下一状态候选集合不能为空")
        if self.next_candidate_count and len(candidates) != self.next_candidate_count:
            raise ValueError(
                "候选重建数量与回放指纹不一致："
                f"stored={self.next_candidate_count}, rebuilt={len(candidates)}"
            )
        if self.next_candidate_ids_sha256 is not None and (
            _candidate_ids_sha256(candidates) != self.next_candidate_ids_sha256
        ):
            raise ValueError("重建的下一状态候选 ID 与回放指纹不一致")
        if any(
            item.action.source_state_fingerprint != str(self.next_state_fingerprint)
            for item in candidates
        ):
            raise ValueError("重建的下一状态候选不属于记录的下一状态")
        if len({item.action.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("重建的下一状态候选包含重复动作 ID")
        return candidates

    def next_candidate_feature_matrix(
        self,
        candidate_rebuilder: CandidateRebuilder | None = None,
    ) -> np.ndarray:
        candidates = self.materialize_next_candidates(candidate_rebuilder)
        if not candidates:
            return np.empty((0, 0), dtype=np.float32)
        matrix = np.asarray([item.features for item in candidates], dtype=np.float32)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("重建的下一状态候选特征包含非有限值")
        return matrix

    def to_compact_dict(self) -> dict[str, Any]:
        """序列化回放；只保存一份状态/预测上下文，不保存候选特征矩阵。"""

        return {
            "state_features": list(self.state_features),
            "action": self.action.to_dict(),
            "action_features": list(self.action_features),
            "reward": self.reward,
            "next_state_features": (
                None
                if self.next_state_features is None
                else list(self.next_state_features)
            ),
            "next_state_fingerprint": self.next_state_fingerprint,
            "next_candidate_rebuild_context_zlib": (
                self.next_candidate_rebuild_context_zlib
            ),
            "next_candidate_count": int(self.next_candidate_count),
            "next_candidate_ids_sha256": self.next_candidate_ids_sha256,
            "current_candidate_rebuild_context_zlib": self.current_candidate_rebuild_context_zlib,
            "current_candidate_count": int(self.current_candidate_count),
            "current_candidate_ids_sha256": self.current_candidate_ids_sha256,
            "terminated": self.terminated,
            "duration_slots": self.duration_slots,
        }

    @classmethod
    def from_compact_dict(cls, value: Mapping[str, Any]) -> "ReplayTransition":
        """恢复新格式经验；旧的全候选特征矩阵格式明确拒绝。"""

        if "next_candidate_rebuild_context_zlib" not in value:
            raise ValueError(
                "Checkpoint Replay Buffer 使用旧的完整候选矩阵格式，拒绝恢复"
            )
        if (
            value.get("current_candidate_rebuild_context_zlib") is None
            or int(value.get("current_candidate_count", 0)) <= 0
            or not value.get("current_candidate_ids_sha256")
        ):
            raise ValueError("旧回放缺少当前合法候选重建合同，拒绝恢复")
        next_state_raw = value.get("next_state_features")
        done = bool(value["terminated"])
        reward = float(value["reward"])
        duration_slots = int(value["duration_slots"])
        next_fingerprint = value.get("next_state_fingerprint")
        context = value.get("next_candidate_rebuild_context_zlib")
        candidate_count = int(value.get("next_candidate_count", 0))
        candidate_ids_sha256 = value.get("next_candidate_ids_sha256")
        if not np.isfinite(reward) or duration_slots <= 0:
            raise ValueError("回放奖励或持续 Slot 数无效")
        if done:
            if next_state_raw is not None or next_fingerprint is not None or context is not None:
                raise ValueError("终止转移不得带有下一状态候选上下文")
            if candidate_count != 0 or candidate_ids_sha256 is not None:
                raise ValueError("终止转移不得声明下一候选数量或指纹")
        elif (
            next_state_raw is None
            or not next_fingerprint
            or context is None
            or candidate_count <= 0
            or not candidate_ids_sha256
        ):
            raise ValueError("非终止转移必须保留完整下一状态候选重建合同")
        return cls(
            state_features=_vector_tuple(value["state_features"], "state_features"),
            action=ActionCandidate.from_dict(value["action"]),
            action_features=_vector_tuple(value["action_features"], "action_features"),
            reward=reward,
            next_state_features=(
                None
                if next_state_raw is None
                else _vector_tuple(next_state_raw, "next_state_features")
            ),
            next_state_fingerprint=None if done else str(next_fingerprint),
            next_legal_candidates=(),
            terminated=done,
            duration_slots=duration_slots,
            next_candidate_rebuild_context_zlib=(
                None if context is None else bytes(context)
            ),
            next_candidate_count=candidate_count,
            next_candidate_ids_sha256=(
                None if candidate_ids_sha256 is None else str(candidate_ids_sha256)
            ),
            current_candidate_rebuild_context_zlib=bytes(value["current_candidate_rebuild_context_zlib"]),
            current_candidate_count=int(value["current_candidate_count"]),
            current_candidate_ids_sha256=str(value["current_candidate_ids_sha256"]),
        )


class ReplayBuffer:
    """固定容量环形回放，不按经验重复保存完整候选特征矩阵。"""

    STATE_SCHEMA_VERSION = 6

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
        # 相邻转移共享同一状态上下文，避免分别保存当前和下一状态的重复字节。
        self._context_pool: dict[str, bytes] = {}
        self._context_refcounts: dict[str, int] = {}

    def _pack_context(
        self, context: Mapping[str, Any], state_features: Any, fingerprint: str
    ) -> bytes:
        if context.get("source_state_fingerprint") != fingerprint:
            raise ValueError("候选上下文与经验状态指纹不一致")
        compact = dict(context)
        recorded = np.asarray(compact.pop("state_features", None), dtype=np.float32).reshape(-1)
        if not np.array_equal(recorded, np.asarray(state_features, dtype=np.float32)):
            raise ValueError("候选上下文与经验状态特征不一致")
        blob = zlib.compress(pickle.dumps(compact, protocol=5), level=1)
        existing = self._context_pool.get(fingerprint)
        if existing is not None and existing != blob:
            raise ValueError("同一状态指纹对应不同候选上下文")
        return existing if existing is not None else blob

    def _retain_context(self, fingerprint: str, blob: bytes) -> bytes:
        existing = self._context_pool.setdefault(fingerprint, blob)
        if existing != blob:
            raise ValueError("相同状态的压缩候选上下文不一致")
        self._context_refcounts[fingerprint] = self._context_refcounts.get(fingerprint, 0) + 1
        return existing

    def _release_contexts(self, transition: ReplayTransition) -> None:
        fingerprints = [transition.action.source_state_fingerprint]
        if not transition.terminated:
            fingerprints.append(str(transition.next_state_fingerprint))
        for fingerprint in fingerprints:
            remaining = self._context_refcounts[fingerprint] - 1
            if remaining:
                self._context_refcounts[fingerprint] = remaining
            else:
                del self._context_refcounts[fingerprint]
                del self._context_pool[fingerprint]

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        state_features: Any,
        action: ActionCandidate,
        action_features: Any,
        current_legal_candidates: Sequence[CandidateOption],
        current_candidate_rebuild_context: Mapping[str, Any],
        reward: float,
        next_state_features: Any | None,
        next_state_fingerprint: str | None,
        next_legal_candidates: Sequence[CandidateOption],
        next_candidate_rebuild_context: Mapping[str, Any] | None,
        terminated: bool,
        duration_slots: int = 1,
    ) -> ReplayTransition:
        if not np.isfinite(float(reward)):
            raise ValueError("经验奖励必须为有限数值")
        if int(duration_slots) <= 0:
            raise ValueError("动作持续 Slot 数必须为正整数")
        current_candidates = tuple(current_legal_candidates)
        source_fingerprint = action.source_state_fingerprint
        if not current_candidates or any(
            item.action.source_state_fingerprint != source_fingerprint
            for item in current_candidates
        ):
            raise ValueError("当前合法候选缺失或不属于所选动作状态")
        selected = next(
            (item for item in current_candidates
             if item.action.candidate_id == action.candidate_id),
            None,
        )
        if selected is None or selected.action.to_dict() != action.to_dict() or not np.array_equal(
            np.asarray(selected.features, dtype=np.float32), np.asarray(action_features, dtype=np.float32)
        ):
            raise ValueError("回放所选动作/特征与当前合法候选不一致")
        current_context_zlib = self._pack_context(
            current_candidate_rebuild_context, state_features, source_fingerprint
        )
        candidates = tuple(next_legal_candidates)
        if terminated:
            if (
                next_state_features is not None
                or next_state_fingerprint is not None
                or candidates
                or next_candidate_rebuild_context is not None
            ):
                raise ValueError("终止转移不得提供下一状态或下一候选上下文")
            next_state = None
            context_zlib = None
            candidate_count = 0
            candidate_ids_sha256 = None
        else:
            if (
                next_state_features is None
                or not next_state_fingerprint
                or not candidates
                or next_candidate_rebuild_context is None
            ):
                raise ValueError("非终止转移必须提供下一观测、候选及可重建上下文")
            next_state = _vector_tuple(next_state_features, "next_state_features")
            if any(not np.all(np.isfinite(np.asarray(item.features))) for item in candidates):
                raise ValueError("下一状态候选特征必须为有限数值")
            if any(
                item.action.source_state_fingerprint != str(next_state_fingerprint)
                for item in candidates
            ):
                raise ValueError("下一状态候选与经验中的下一状态指纹不一致")
            if next_candidate_rebuild_context.get("source_state_fingerprint") != str(
                next_state_fingerprint
            ):
                raise ValueError("候选重建上下文与经验中的下一状态指纹不一致")
            context_zlib = self._pack_context(
                next_candidate_rebuild_context, next_state, str(next_state_fingerprint)
            )
            candidate_count = len(candidates)
            candidate_ids_sha256 = _candidate_ids_sha256(candidates)
        if not action.source_state_fingerprint:
            raise ValueError("回放动作缺少其来源状态指纹")
        current_ids_sha256 = _candidate_ids_sha256(current_candidates)
        if not terminated and source_fingerprint == str(next_state_fingerprint):
            if (
                current_context_zlib != context_zlib
                or current_ids_sha256 != candidate_ids_sha256
            ):
                raise ValueError("相同当前/下一状态身份对应不一致的候选合同")
        transition = ReplayTransition(
            state_features=_vector_tuple(state_features, "state_features"),
            action=ActionCandidate.from_dict(action.to_dict()),
            action_features=_vector_tuple(action_features, "action_features"),
            reward=float(reward),
            next_state_features=next_state,
            next_state_fingerprint=(None if terminated else str(next_state_fingerprint)),
            next_legal_candidates=(),
            terminated=bool(terminated),
            duration_slots=int(duration_slots),
            next_candidate_rebuild_context_zlib=context_zlib,
            next_candidate_count=candidate_count,
            next_candidate_ids_sha256=candidate_ids_sha256,
            current_candidate_rebuild_context_zlib=current_context_zlib,
            current_candidate_count=len(current_candidates),
            current_candidate_ids_sha256=current_ids_sha256,
        )
        if len(self._items) == self.capacity:
            self._release_contexts(self._items[self._position])
        transition = replace(
            transition,
            current_candidate_rebuild_context_zlib=self._retain_context(source_fingerprint, current_context_zlib),
            next_candidate_rebuild_context_zlib=(
                None if terminated else self._retain_context(str(next_state_fingerprint), context_zlib)
            ),
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
            "schema_version": self.STATE_SCHEMA_VERSION,
            "capacity": self.capacity,
            "position": self._position,
            "items": [item.to_compact_dict() for item in self._items],
            "rng_state": self._rng.getstate(),
            "write_count": self.write_count,
            "sample_calls": self.sample_calls,
            "sample_count": self.sample_count,
            "candidate_feature_matrices_persisted": False,
            "unique_candidate_context_count": len(self._context_pool),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError("Checkpoint Replay Buffer 格式过旧或不兼容，拒绝恢复")
        if bool(state.get("candidate_feature_matrices_persisted", True)):
            raise ValueError("Checkpoint Replay Buffer 保存了旧的候选特征矩阵")
        capacity = int(state["capacity"])
        items = [ReplayTransition.from_compact_dict(item) for item in state["items"]]
        if capacity != self.capacity or len(items) > self.capacity:
            raise ValueError("回放状态容量与当前缓冲区不匹配")
        context_pool: dict[str, bytes] = {}
        context_refcounts: dict[str, int] = {}

        def intern(fingerprint: str, blob: bytes | None) -> bytes:
            if not fingerprint or blob is None:
                raise ValueError("Checkpoint 状态身份或候选上下文缺失")
            existing = context_pool.setdefault(fingerprint, blob)
            if existing != blob:
                raise ValueError("Checkpoint 相同状态对应不同候选上下文")
            context_refcounts[fingerprint] = context_refcounts.get(fingerprint, 0) + 1
            return existing

        rebuilt_items = [
            replace(
                item,
                current_candidate_rebuild_context_zlib=intern(
                    item.action.source_state_fingerprint, item.current_candidate_rebuild_context_zlib
                ),
                next_candidate_rebuild_context_zlib=(
                    None if item.terminated else intern(
                        str(item.next_state_fingerprint), item.next_candidate_rebuild_context_zlib
                    )
                ),
            )
            for item in items
        ]
        restored_rng = random.Random()
        restored_rng.setstate(state["rng_state"])
        restored_position = int(state["position"])
        if not 0 <= restored_position < self.capacity:
            raise ValueError("Checkpoint Replay Buffer 环形写入位置非法")
        # 所有身份、上下文与随机状态校验通过后，才替换当前回放状态。
        self._items = rebuilt_items
        self._context_pool = context_pool
        self._context_refcounts = context_refcounts
        self._position = restored_position
        self._rng = restored_rng
        self.write_count = int(state.get("write_count", len(items)))
        self.sample_calls = int(state.get("sample_calls", 0))
        self.sample_count = int(state.get("sample_count", 0))


__all__ = ["ReplayTransition", "ReplayBuffer"]
