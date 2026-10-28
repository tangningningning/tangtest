"""保存转移及按 Episode 身份重建的当前/下一状态候选上下文。"""

from __future__ import annotations

import hashlib
import pickle
import random
import zlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .candidate_features import ActionCandidate, CandidateOption


def _vector_tuple(value: Any, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须为有限数值")
    return tuple(float(item) for item in array)


def _state_vector(value: Any, name: str, *, copy: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须为有限数值")
    if copy:
        array = array.copy()
    if array.flags.writeable:
        array.setflags(write=False)
    return array


def _candidate_ids_sha256(candidates: Sequence[CandidateOption]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.action.candidate_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_operation_identity(
    candidates: Sequence[CandidateOption],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for operation in _ACTION_OPERATIONS:
        selected = tuple(
            item for item in candidates if item.action.operation == operation
        )
        result[operation] = {
            "count": len(selected),
            "ids_sha256": _candidate_ids_sha256(selected),
        }
    return result


def _validate_candidate_operation_context(
    context: Mapping[str, Any],
    candidates: Sequence[CandidateOption],
    label: str,
) -> None:
    recorded = context.get("candidate_identity_by_operation")
    expected = _candidate_operation_identity(candidates)
    if recorded != expected:
        raise ValueError(f"{label}候选类别数量/ID 顺序身份不一致")


def _validate_candidate_context(
    context: Mapping[str, Any], candidates: Sequence[CandidateOption], label: str
) -> None:
    if int(context.get("candidate_count", -1)) != len(candidates):
        raise ValueError(f"{label}候选数量与重建上下文不一致")
    recorded_ids = context.get("candidate_ids_sha256")
    if recorded_ids != _candidate_ids_sha256(candidates):
        raise ValueError(f"{label}候选 ID 顺序与重建上下文不一致")


_ACTION_OPERATIONS = ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
_BATCH_REUSE_MAX_CONTEXTS = 16
_BATCH_REUSE_MAX_CANDIDATES = 4096
_VERIFIED_CONTEXT_MARKER_KEY = "_replay_verified_candidate_payload_digest"
_VERIFIED_CONTEXT_MARKER = object()
_VERIFIED_CONTEXT_DIGEST_CACHE_MAX_ENTRIES = 128
_VERIFIED_CONTEXT_DIGEST_CACHE_MAX_BYTES = 16 * 1024 * 1024


def _candidate_context_cache_key(context: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(context.get("episode_rebuild_id", "")),
        str(context.get("episode_rebuild_spec_sha256", "")),
        str(context.get("candidate_rebuild_payload_sha256", "")),
        str(context.get("source_state_fingerprint", "")),
        str(context.get("state_feature_contract_version", "")),
        str(context.get("candidate_feature_contract_version", "")),
        str(context.get("network_architecture_version", "")),
    )


def _context_from_blob(
    blob: bytes,
    state_features: Any,
    fingerprint: str,
    label: str,
    *,
    verified_payload_digests: OrderedDict[bytes, str] | None = None,
) -> Mapping[str, Any]:
    context = pickle.loads(zlib.decompress(blob))
    if not isinstance(context, Mapping):
        raise ValueError(f"{label}候选重建上下文必须是映射")
    if context.get("source_state_fingerprint") != str(fingerprint):
        raise ValueError(f"{label}候选上下文与状态指纹不匹配")
    if "state_features" in context:
        raise ValueError("紧凑回放上下文不得重复保存状态特征")
    expanded = dict(context)
    expanded["state_features"] = np.asarray(state_features, dtype=np.float32)
    verified_digest: str | None = None
    if verified_payload_digests is not None:
        verified_digest = verified_payload_digests.get(blob)
        if verified_digest is not None:
            verified_payload_digests.move_to_end(blob)
            if expanded.get("candidate_rebuild_payload_sha256") != verified_digest:
                raise ValueError("回放候选上下文摘要与已验证的 Replay 字节块不一致")
    if verified_digest is None:
        from .environment_adapter import _candidate_rebuild_payload_sha256

        verified_digest = _candidate_rebuild_payload_sha256(expanded)
        if expanded.get("candidate_rebuild_payload_sha256") != verified_digest:
            raise ValueError("回放候选重建上下文摘要不匹配")
        if verified_payload_digests is not None:
            _remember_verified_context_payload_digest(
                verified_payload_digests, blob, verified_digest
            )
    expanded[_VERIFIED_CONTEXT_MARKER_KEY] = (
        _VERIFIED_CONTEXT_MARKER,
        verified_digest,
    )
    return MappingProxyType(expanded)


def _remember_verified_context_payload_digest(
    cache: OrderedDict[bytes, str], blob: bytes, digest: str
) -> None:
    existing = cache.pop(blob, None)
    if existing is not None and existing != digest:
        raise ValueError("相同 Replay 上下文字节块对应不同候选摘要")
    if existing is None:
        cache[blob] = str(digest)
        retained_bytes = sum(len(item) for item in cache)
    else:
        cache[blob] = existing
        retained_bytes = sum(len(item) for item in cache)
    while (
        len(cache) > _VERIFIED_CONTEXT_DIGEST_CACHE_MAX_ENTRIES
        or retained_bytes > _VERIFIED_CONTEXT_DIGEST_CACHE_MAX_BYTES
    ):
        evicted_blob, _ = cache.popitem(last=False)
        retained_bytes -= len(evicted_blob)


def _verified_candidate_rebuild_payload_digest(
    context: Mapping[str, Any],
) -> str | None:
    marker = context.get(_VERIFIED_CONTEXT_MARKER_KEY)
    if (
        isinstance(marker, tuple)
        and len(marker) == 2
        and marker[0] is _VERIFIED_CONTEXT_MARKER
        and isinstance(marker[1], str)
    ):
        return marker[1]
    return None


def _validate_candidate_operation_subset(
    candidates: Sequence[CandidateOption],
    context: Mapping[str, Any],
    operation: str,
    fingerprint: str,
) -> None:
    identities = context.get("candidate_identity_by_operation")
    if not isinstance(identities, Mapping) or operation not in identities:
        raise ValueError("回放上下文缺少当前动作类别的候选身份")
    expected = identities[operation]
    if not isinstance(expected, Mapping):
        raise ValueError("当前动作类别候选身份无效")
    if len(candidates) != int(expected.get("count", -1)):
        raise ValueError("当前动作类别候选数量与上下文不一致")
    if _candidate_ids_sha256(candidates) != expected.get("ids_sha256"):
        raise ValueError("当前动作类别候选 ID/顺序与上下文不一致")
    if any(
        item.action.operation != operation
        or item.action.source_state_fingerprint != str(fingerprint)
        or not np.isfinite(np.asarray(item.features, dtype=np.float32)).all()
        for item in candidates
    ):
        raise ValueError("当前动作类别候选包含非法动作、状态或特征")
    if len({item.action.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("当前动作类别候选包含重复动作 ID")


class ReplayCandidateBatchReuseCache:
    """只暂存本 Batch 内重复语义请求的有界候选集合。"""

    def __init__(self, reusable_context_keys: set[tuple[str, ...]]) -> None:
        self._reusable_context_keys = frozenset(reusable_context_keys)
        self._items: OrderedDict[tuple[str, ...], tuple[CandidateOption, ...]] = OrderedDict()
        self._candidate_count = 0
        self._stats = {"hit_count": 0, "retained_context_count": 0, "retained_candidate_count": 0,
                       "eviction_count": 0, "bypass_count": 0}

    @staticmethod
    def key(
        context: Mapping[str, Any], *, operation_filter: str | None = None
    ) -> tuple[str, ...]:
        scope = "ALL" if operation_filter is None else str(operation_filter)
        return (*_candidate_context_cache_key(context), scope)

    @staticmethod
    def key_from_context_key(
        context_key: tuple[str, ...], *, operation_filter: str | None = None
    ) -> tuple[str, ...]:
        scope = "ALL" if operation_filter is None else str(operation_filter)
        return (*context_key, scope)

    def get(
        self,
        context: Mapping[str, Any],
        *,
        operation_filter: str | None = None,
    ) -> tuple[CandidateOption, ...] | None:
        key = self.key(context, operation_filter=operation_filter)
        candidates = (
            self._items.get(key)
            if key in self._reusable_context_keys
            else None
        )
        if candidates is None and operation_filter is not None:
            # A complete next-state set is an exact superset of a current
            # operation subset for the same semantic context. The adapter
            # filters it and applies the ordinary identity checks afterward.
            full_key = self.key(context)
            if full_key in self._reusable_context_keys:
                candidates = self._items.get(full_key)
                if candidates is not None:
                    key = full_key
        if candidates is not None:
            self._items.move_to_end(key)
            self._stats["hit_count"] += 1
        return candidates

    def put(
        self,
        context: Mapping[str, Any],
        candidates: Sequence[CandidateOption],
        *,
        operation_filter: str | None = None,
    ) -> None:
        key = self.key(context, operation_filter=operation_filter)
        if key not in self._reusable_context_keys or key in self._items:
            return
        candidate_tuple = tuple(candidates)
        if len(candidate_tuple) > _BATCH_REUSE_MAX_CANDIDATES:
            self._stats["bypass_count"] += 1
            return
        self._items[key] = candidate_tuple
        self._candidate_count += len(candidate_tuple)
        while (
            len(self._items) > _BATCH_REUSE_MAX_CONTEXTS
            or self._candidate_count > _BATCH_REUSE_MAX_CANDIDATES
        ):
            _, evicted = self._items.popitem(last=False)
            self._candidate_count -= len(evicted)
            self._stats["eviction_count"] += 1
        self._stats["retained_context_count"] = len(self._items)
        self._stats["retained_candidate_count"] = self._candidate_count

    @property
    def metrics(self) -> dict[str, int]:
        return {
            **self._stats,
            "eligible_context_count": len(
                {key[:-1] for key in self._reusable_context_keys}
            ),
            "eligible_scope_count": len(self._reusable_context_keys),
            "max_context_count": _BATCH_REUSE_MAX_CONTEXTS,
            "max_candidate_count": _BATCH_REUSE_MAX_CANDIDATES,
        }


CandidateRebuilder = Callable[[Mapping[str, Any]], Sequence[CandidateOption]]


@dataclass(frozen=True)
class ReplayTransition:
    """经验只保留所选动作及当前/下一状态重建上下文，不保存候选特征矩阵。"""

    state_features: np.ndarray
    action: ActionCandidate
    action_features: tuple[float, ...]
    reward: float
    next_state_features: np.ndarray | None
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

    def candidate_rebuild_context_key(self, *, current: bool) -> tuple[str, ...] | None:
        blob = (
            self.current_candidate_rebuild_context_zlib
            if current
            else self.next_candidate_rebuild_context_zlib
        )
        if blob is None:
            return None
        context = pickle.loads(zlib.decompress(blob))
        if not isinstance(context, Mapping):
            raise ValueError("回放候选重建上下文必须是映射")
        return _candidate_context_cache_key(context)

    def materialize_current_candidates(
        self,
        candidate_rebuilder: CandidateRebuilder | None,
        *,
        batch_cache: ReplayCandidateBatchReuseCache | None = None,
    ) -> tuple[CandidateOption, ...]:
        """精确重建已执行动作类别，供 Dueling 类别内优势均值使用。"""

        fingerprint = self.action.source_state_fingerprint
        operation = self.action.operation
        if self.current_legal_candidates:
            full_candidates = tuple(self.current_legal_candidates)
            if (
                len(full_candidates) != self.current_candidate_count
                or _candidate_ids_sha256(full_candidates) != self.current_candidate_ids_sha256
            ):
                raise ValueError("当前状态候选与 Replay 完整候选身份不一致")
            candidates = tuple(
                item for item in full_candidates if item.action.operation == operation
            )
            _validate_candidate_operation_subset(
                candidates,
                {"candidate_identity_by_operation": _candidate_operation_identity(full_candidates)},
                operation,
                fingerprint,
            )
            return candidates

        if self.current_candidate_rebuild_context_zlib is None:
            raise ValueError("回放转移缺少当前候选重建上下文")
        owner = getattr(candidate_rebuilder, "__self__", None)
        context = _context_from_blob(
            self.current_candidate_rebuild_context_zlib,
            self.state_features,
            fingerprint,
            "当前状态",
            verified_payload_digests=getattr(
                owner, "_verified_candidate_payload_digests", None
            ),
        )
        if (
            int(context.get("candidate_count", -1)) != self.current_candidate_count
            or str(context.get("candidate_ids_sha256", ""))
            != str(self.current_candidate_ids_sha256)
        ):
            raise ValueError("当前 Replay 与重建上下文的完整候选身份不一致")
        if candidate_rebuilder is None:
            raise ValueError("回放转移缺少当前候选重建器")
        operation_rebuilder = getattr(owner, "rebuild_candidates_for_operation", None)
        if callable(operation_rebuilder):
            candidates = tuple(
                operation_rebuilder(context, operation, batch_cache=batch_cache)
            )
        else:
            full_candidates = tuple(candidate_rebuilder(context))
            if (
                len(full_candidates) != self.current_candidate_count
                or _candidate_ids_sha256(full_candidates) != self.current_candidate_ids_sha256
            ):
                raise ValueError("当前状态完整候选与 Replay 身份不一致")
            candidates = tuple(
                item for item in full_candidates if item.action.operation == operation
            )
        _validate_candidate_operation_subset(
            candidates, context, operation, fingerprint
        )
        return candidates

    def materialize_next_candidates(
        self,
        candidate_rebuilder: CandidateRebuilder | None,
        *,
        batch_cache: ReplayCandidateBatchReuseCache | None = None,
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
            if self.next_state_features is None:
                raise ValueError("候选重建缺少已保存的下一状态特征")
            owner = getattr(candidate_rebuilder, "__self__", None)
            context = _context_from_blob(
                self.next_candidate_rebuild_context_zlib,
                self.next_state_features,
                str(self.next_state_fingerprint),
                "下一状态",
                verified_payload_digests=getattr(
                    owner, "_verified_candidate_payload_digests", None
                ),
            )
            if (
                int(context.get("candidate_count", -1)) != self.next_candidate_count
                or str(context.get("candidate_ids_sha256", ""))
                != str(self.next_candidate_ids_sha256)
            ):
                raise ValueError("下一 Replay 与重建上下文的完整候选身份不一致")
            replay_rebuilder = getattr(owner, "rebuild_candidates", None)
            if batch_cache is not None and callable(replay_rebuilder):
                candidates = tuple(replay_rebuilder(context, batch_cache=batch_cache))
            else:
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
            "state_features": _state_vector(
                self.state_features, "state_features", copy=False
            ),
            "action": self.action.to_dict(),
            "action_features": list(self.action_features),
            "reward": self.reward,
            "next_state_features": (
                None
                if self.next_state_features is None
                else _state_vector(
                    self.next_state_features, "next_state_features", copy=False
                )
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
            state_features=_state_vector(
                value["state_features"], "state_features", copy=False
            ),
            action=ActionCandidate.from_dict(value["action"]),
            action_features=_vector_tuple(value["action_features"], "action_features"),
            reward=reward,
            next_state_features=(
                None
                if next_state_raw is None
                else _state_vector(
                    next_state_raw, "next_state_features", copy=False
                )
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

    STATE_SCHEMA_VERSION = 12

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
        self._state_feature_pool: dict[str, np.ndarray] = {}
        self._state_feature_refcounts: dict[str, int] = {}
        # 相邻转移共享同一状态上下文，避免分别保存当前和下一状态的重复字节。
        self._context_pool: dict[str, bytes] = {}
        self._context_refcounts: dict[str, int] = {}
        self._verified_candidate_payload_digests: OrderedDict[bytes, str] = (
            OrderedDict()
        )
        self._candidate_rebuilders: dict[str, Any] = {}
        self._episode_rebuild_refcounts: dict[str, int] = {}

    def register_candidate_rebuilder(self, rebuilder: Any) -> None:
        """登记与历史 Episode 绑定的轻量候选重建器。"""

        episode_id = str(getattr(rebuilder, "_episode_rebuild_id", ""))
        spec_sha256 = str(getattr(rebuilder, "_episode_rebuild_spec_sha256", ""))
        if not episode_id or len(spec_sha256) != 64:
            raise ValueError("Episode 候选重建器缺少有效身份合同")
        try:
            bytes.fromhex(episode_id)
            bytes.fromhex(spec_sha256)
        except ValueError:
            raise ValueError("Episode 重建身份摘要不是有效十六进制") from None
        from .environment_adapter import episode_rebuild_identity

        expected_episode_id = episode_rebuild_identity(
            getattr(rebuilder, "episode_seed", None), spec_sha256
        )
        if episode_id != expected_episode_id:
            raise ValueError("Episode 重建器身份与种子/规格摘要不匹配")
        existing = self._candidate_rebuilders.get(episode_id)
        if existing is not None:
            if str(existing._episode_rebuild_spec_sha256) != spec_sha256:
                raise ValueError("相同 Episode 重建身份对应不同规格")
            return
        self._candidate_rebuilders[episode_id] = rebuilder
        self._episode_rebuild_refcounts.setdefault(episode_id, 0)

    @staticmethod
    def _context_episode_identity(blob: bytes) -> tuple[str, str]:
        context = pickle.loads(zlib.decompress(blob))
        if not isinstance(context, Mapping):
            raise ValueError("Replay 候选上下文必须是映射")
        episode_id = str(context.get("episode_rebuild_id", ""))
        spec_sha256 = str(context.get("episode_rebuild_spec_sha256", ""))
        if not episode_id or len(spec_sha256) != 64:
            raise ValueError("Replay 候选上下文缺少 Episode 重建身份")
        return episode_id, spec_sha256

    def _validate_registered_context(self, blob: bytes) -> tuple[str, Any]:
        episode_id, spec_sha256 = self._context_episode_identity(blob)
        rebuilder = self._candidate_rebuilders.get(episode_id)
        if rebuilder is None:
            raise ValueError("Replay 中的 Episode 尚未登记候选重建器")
        if str(rebuilder._episode_rebuild_spec_sha256) != spec_sha256:
            raise ValueError("Replay 上下文与 Episode 重建器规格不匹配")
        return episode_id, rebuilder

    def rebuild_candidates(
        self,
        context: Mapping[str, Any],
        *,
        batch_cache: ReplayCandidateBatchReuseCache | None = None,
    ) -> Sequence[CandidateOption]:
        """按上下文所属 Episode 分派，禁止用当前 Episode 映射重建历史经验。"""

        episode_id = str(context.get("episode_rebuild_id", ""))
        spec_sha256 = str(context.get("episode_rebuild_spec_sha256", ""))
        rebuilder = self._candidate_rebuilders.get(episode_id)
        if rebuilder is None:
            raise ValueError("Replay 缺少该历史 Episode 的候选重建器")
        if str(rebuilder._episode_rebuild_spec_sha256) != spec_sha256:
            raise ValueError("Replay 候选上下文与历史 Episode 规格不匹配")
        return rebuilder.rebuild_candidates(context, batch_cache=batch_cache)

    def rebuild_candidates_for_operation(
        self,
        context: Mapping[str, Any],
        operation: str,
        *,
        batch_cache: ReplayCandidateBatchReuseCache | None = None,
    ) -> Sequence[CandidateOption]:
        """按 Episode 身份只重建一个当前动作类别。"""

        episode_id = str(context.get("episode_rebuild_id", ""))
        spec_sha256 = str(context.get("episode_rebuild_spec_sha256", ""))
        rebuilder = self._candidate_rebuilders.get(episode_id)
        if rebuilder is None:
            raise ValueError("Replay 缺少该历史 Episode 的候选重建器")
        if str(rebuilder._episode_rebuild_spec_sha256) != spec_sha256:
            raise ValueError("Replay 候选上下文与历史 Episode 规格不匹配")
        return rebuilder.rebuild_candidates_for_operation(
            context, operation, batch_cache=batch_cache
        )

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
        episode_id, _ = self._validate_registered_context(blob)
        existing = self._context_pool.setdefault(fingerprint, blob)
        if existing != blob:
            raise ValueError("相同状态的压缩候选上下文不一致")
        self._context_refcounts[fingerprint] = self._context_refcounts.get(fingerprint, 0) + 1
        self._episode_rebuild_refcounts[episode_id] = self._episode_rebuild_refcounts.get(
            episode_id, 0
        ) + 1
        return existing

    def _release_contexts(self, transition: ReplayTransition) -> None:
        fingerprints = [transition.action.source_state_fingerprint]
        if not transition.terminated:
            fingerprints.append(str(transition.next_state_fingerprint))
        for fingerprint in fingerprints:
            remaining = self._context_refcounts[fingerprint] - 1
            episode_id, _ = self._context_episode_identity(self._context_pool[fingerprint])
            episode_remaining = self._episode_rebuild_refcounts[episode_id] - 1
            if episode_remaining:
                self._episode_rebuild_refcounts[episode_id] = episode_remaining
            else:
                del self._episode_rebuild_refcounts[episode_id]
            if remaining:
                self._context_refcounts[fingerprint] = remaining
            else:
                del self._context_refcounts[fingerprint]
                del self._context_pool[fingerprint]

    def _validate_state_feature_pairs(
        self, pairs: Sequence[tuple[str, np.ndarray]]
    ) -> None:
        pending: dict[str, np.ndarray] = {}
        for fingerprint, values in pairs:
            key = str(fingerprint)
            if not key:
                raise ValueError("Replay 状态向量缺少状态指纹")
            existing = self._state_feature_pool.get(key, pending.get(key))
            if existing is not None and not np.array_equal(existing, values):
                raise ValueError("相同 Replay 状态指纹对应不同状态向量")
            pending[key] = values if existing is None else existing

    def _retain_state_feature_pairs(
        self, pairs: Sequence[tuple[str, np.ndarray]]
    ) -> tuple[np.ndarray, ...]:
        self._validate_state_feature_pairs(pairs)
        retained: list[np.ndarray] = []
        for fingerprint, values in pairs:
            key = str(fingerprint)
            existing = self._state_feature_pool.get(key)
            if existing is None:
                existing = values
                self._state_feature_pool[key] = existing
            self._state_feature_refcounts[key] = (
                self._state_feature_refcounts.get(key, 0) + 1
            )
            retained.append(existing)
        return tuple(retained)

    def _release_state_features(self, transition: ReplayTransition) -> None:
        fingerprints = [transition.action.source_state_fingerprint]
        if not transition.terminated:
            if transition.next_state_fingerprint is None:
                raise ValueError("Replay 非终止转移缺少下一状态指纹")
            fingerprints.append(str(transition.next_state_fingerprint))
        for fingerprint in fingerprints:
            remaining = self._state_feature_refcounts[fingerprint] - 1
            if remaining:
                self._state_feature_refcounts[fingerprint] = remaining
            else:
                del self._state_feature_refcounts[fingerprint]
                del self._state_feature_pool[fingerprint]

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
        _validate_candidate_context(
            current_candidate_rebuild_context, current_candidates, "当前"
        )
        _validate_candidate_operation_context(
            current_candidate_rebuild_context, current_candidates, "当前"
        )
        current_context_zlib = self._pack_context(
            current_candidate_rebuild_context, state_features, source_fingerprint
        )
        self._validate_registered_context(current_context_zlib)
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
            next_state = _state_vector(next_state_features, "next_state_features")
            if any(not np.all(np.isfinite(np.asarray(item.features))) for item in candidates):
                raise ValueError("下一状态候选特征必须为有限数值")
            if any(
                item.action.source_state_fingerprint != str(next_state_fingerprint)
                for item in candidates
            ):
                raise ValueError("下一状态候选与经验中的下一状态指纹不一致")
            _validate_candidate_context(
                next_candidate_rebuild_context, candidates, "下一状态"
            )
            _validate_candidate_operation_context(
                next_candidate_rebuild_context, candidates, "下一状态"
            )
            if next_candidate_rebuild_context.get("source_state_fingerprint") != str(
                next_state_fingerprint
            ):
                raise ValueError("候选重建上下文与经验中的下一状态指纹不一致")
            context_zlib = self._pack_context(
                next_candidate_rebuild_context, next_state, str(next_state_fingerprint)
            )
            self._validate_registered_context(context_zlib)
            candidate_count = len(candidates)
            candidate_ids_sha256 = _candidate_ids_sha256(candidates)
        if not action.source_state_fingerprint:
            raise ValueError("回放动作缺少其来源状态指纹")
        state_vector = _state_vector(state_features, "state_features")
        state_feature_pairs = [(source_fingerprint, state_vector)]
        if not terminated:
            state_feature_pairs.append(
                (str(next_state_fingerprint), next_state)
            )
        self._validate_state_feature_pairs(state_feature_pairs)
        current_ids_sha256 = _candidate_ids_sha256(current_candidates)
        if not terminated and source_fingerprint == str(next_state_fingerprint):
            if (
                current_context_zlib != context_zlib
                or current_ids_sha256 != candidate_ids_sha256
            ):
                raise ValueError("相同当前/下一状态身份对应不一致的候选合同")
        transition = ReplayTransition(
            state_features=state_vector,
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
            evicted = self._items[self._position]
            self._release_contexts(evicted)
            self._release_state_features(evicted)
        retained_state_vectors = self._retain_state_feature_pairs(
            state_feature_pairs
        )
        transition = replace(
            transition,
            state_features=retained_state_vectors[0],
            next_state_features=(
                None if terminated else retained_state_vectors[1]
            ),
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
        for episode_id, references in tuple(self._episode_rebuild_refcounts.items()):
            if references == 0:
                self._episode_rebuild_refcounts.pop(episode_id, None)
                self._candidate_rebuilders.pop(episode_id, None)
        return transition

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if int(batch_size) <= 0 or int(batch_size) > len(self._items):
            raise ValueError("批量大小必须在当前回放样本数范围内")
        sampled = self._rng.sample(self._items, int(batch_size))
        self.sample_calls += 1
        self.sample_count += int(batch_size)
        return sampled

    def validate_rebuild_contexts(self) -> None:
        """在恢复训练前校验每个唯一候选上下文，不生成候选或运行系统求解。"""

        state_features_by_fingerprint: dict[str, np.ndarray] = {}
        state_refcounts_by_fingerprint: dict[str, int] = {}
        candidate_identity_by_fingerprint: dict[str, tuple[int, str]] = {}

        def retain_state(
            fingerprint: str,
            values: Any,
            candidate_count: int,
            candidate_ids_sha256: str | None,
        ) -> None:
            state_vector = _state_vector(
                values, "Replay rebuild state_features", copy=False
            )
            existing = state_features_by_fingerprint.get(fingerprint)
            if existing is not None and not np.array_equal(existing, state_vector):
                raise ValueError("Replay 中相同状态指纹对应不同状态特征")
            state_features_by_fingerprint[fingerprint] = state_vector
            state_refcounts_by_fingerprint[fingerprint] = (
                state_refcounts_by_fingerprint.get(fingerprint, 0) + 1
            )
            if int(candidate_count) <= 0 or not candidate_ids_sha256:
                raise ValueError("Replay 状态缺少候选数量或候选 ID 摘要")
            candidate_identity = (int(candidate_count), str(candidate_ids_sha256))
            existing_identity = candidate_identity_by_fingerprint.get(fingerprint)
            if existing_identity is not None and existing_identity != candidate_identity:
                raise ValueError("Replay 中相同状态指纹对应不同候选身份")
            candidate_identity_by_fingerprint[fingerprint] = candidate_identity

        for transition in self._items:
            retain_state(
                transition.action.source_state_fingerprint,
                transition.state_features,
                transition.current_candidate_count,
                transition.current_candidate_ids_sha256,
            )
            if not transition.terminated:
                if transition.next_state_fingerprint is None or transition.next_state_features is None:
                    raise ValueError("Replay 非终止转移缺少重建上下文的状态身份")
                retain_state(
                    str(transition.next_state_fingerprint),
                    transition.next_state_features,
                    transition.next_candidate_count,
                    transition.next_candidate_ids_sha256,
                )
        if (
            set(state_features_by_fingerprint) != set(self._context_pool)
            or set(state_features_by_fingerprint) != set(self._state_feature_pool)
            or state_refcounts_by_fingerprint != self._state_feature_refcounts
        ):
            raise ValueError("Replay 状态特征与唯一候选上下文索引不一致")
        if any(
            not np.array_equal(self._state_feature_pool[fingerprint], values)
            for fingerprint, values in state_features_by_fingerprint.items()
        ):
            raise ValueError("Replay 状态向量池与转移中保存的状态特征不一致")
        for fingerprint, blob in self._context_pool.items():
            context = pickle.loads(zlib.decompress(blob))
            if not isinstance(context, Mapping):
                raise ValueError("Replay 候选上下文必须是映射")
            if context.get("source_state_fingerprint") != fingerprint:
                raise ValueError("Replay 候选上下文与状态指纹索引不一致")
            expected_candidate_identity = candidate_identity_by_fingerprint[fingerprint]
            if (
                int(context.get("candidate_count", -1)),
                str(context.get("candidate_ids_sha256", "")),
            ) != expected_candidate_identity:
                raise ValueError("Replay 上下文与状态记录的候选身份不一致")
            expanded_context = dict(context)
            expanded_context["state_features"] = np.asarray(
                state_features_by_fingerprint[fingerprint], dtype=np.float32
            )
            episode_id = str(expanded_context.get("episode_rebuild_id", ""))
            spec_sha256 = str(
                expanded_context.get("episode_rebuild_spec_sha256", "")
            )
            rebuilder = self._candidate_rebuilders.get(episode_id)
            if rebuilder is None:
                raise ValueError("Checkpoint Replay 缺少历史 Episode 重建器")
            if str(rebuilder._episode_rebuild_spec_sha256) != spec_sha256:
                raise ValueError("Checkpoint Replay Episode 重建规格不匹配")
            rebuilder.validate_candidate_rebuild_context_metadata(expanded_context)
            _remember_verified_context_payload_digest(
                self._verified_candidate_payload_digests,
                blob,
                str(expanded_context["candidate_rebuild_payload_sha256"]),
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "state_feature_storage_contract": "float32_fingerprint_interned_v1",
            "capacity": self.capacity,
            "position": self._position,
            "items": [item.to_compact_dict() for item in self._items],
            "rng_state": self._rng.getstate(),
            "write_count": self.write_count,
            "sample_calls": self.sample_calls,
            "sample_count": self.sample_count,
            "candidate_feature_matrices_persisted": False,
            "unique_candidate_context_count": len(self._context_pool),
            "unique_state_feature_count": len(self._state_feature_pool),
            "episode_candidate_rebuilders": {
                episode_id: rebuilder.state_dict()
                for episode_id, rebuilder in self._candidate_rebuilders.items()
                if self._episode_rebuild_refcounts.get(episode_id, 0) > 0
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError("Checkpoint Replay Buffer 格式过旧或不兼容，拒绝恢复")
        if state.get("state_feature_storage_contract") != (
            "float32_fingerprint_interned_v1"
        ):
            raise ValueError("Checkpoint Replay 状态向量存储合同不兼容，拒绝恢复")
        if bool(state.get("candidate_feature_matrices_persisted", True)):
            raise ValueError("Checkpoint Replay Buffer 保存了旧的候选特征矩阵")

        def nonnegative_integer(field: str, *, default: Any = None) -> int:
            if field in state:
                value = state[field]
            elif default is not None:
                value = default
            else:
                raise ValueError(f"Checkpoint Replay 缺少 {field}")
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"Checkpoint Replay {field} 必须是非负整数")
            result = int(value)
            if result < 0:
                raise ValueError(f"Checkpoint Replay {field} 必须是非负整数")
            return result

        capacity = int(state["capacity"])
        items = [ReplayTransition.from_compact_dict(item) for item in state["items"]]
        if capacity != self.capacity or len(items) > self.capacity:
            raise ValueError("回放状态容量与当前缓冲区不匹配")
        restored_write_count = nonnegative_integer("write_count")
        restored_position = nonnegative_integer("position")
        restored_sample_calls = nonnegative_integer("sample_calls", default=0)
        restored_sample_count = nonnegative_integer("sample_count", default=0)
        if len(items) != min(restored_write_count, self.capacity):
            raise ValueError("Checkpoint Replay 写入计数与经验数量不一致")
        if restored_position != restored_write_count % self.capacity:
            raise ValueError("Checkpoint Replay 写入位置与写入计数不一致")
        raw_rebuilders = state.get("episode_candidate_rebuilders")
        if not isinstance(raw_rebuilders, Mapping):
            raise ValueError("Checkpoint Replay 缺少 Episode 候选重建器注册表")
        from .environment_adapter import EpisodeCandidateRebuildAdapter

        candidate_rebuilders = {
            str(episode_id): EpisodeCandidateRebuildAdapter.from_state_dict(value)
            for episode_id, value in raw_rebuilders.items()
        }
        for episode_id, rebuilder in candidate_rebuilders.items():
            if str(rebuilder._episode_rebuild_id) != episode_id:
                raise ValueError("Checkpoint Replay 重建器身份键不一致")
        context_pool: dict[str, bytes] = {}
        context_refcounts: dict[str, int] = {}
        episode_refcounts: dict[str, int] = {}
        state_feature_pool: dict[str, np.ndarray] = {}
        state_feature_refcounts: dict[str, int] = {}

        def intern(fingerprint: str, blob: bytes | None) -> bytes:
            if not fingerprint or blob is None:
                raise ValueError("Checkpoint 状态身份或候选上下文缺失")
            existing = context_pool.setdefault(fingerprint, blob)
            if existing != blob:
                raise ValueError("Checkpoint 相同状态对应不同候选上下文")
            episode_id, spec_sha256 = self._context_episode_identity(blob)
            rebuilder = candidate_rebuilders.get(episode_id)
            if rebuilder is None:
                raise ValueError("Checkpoint 上下文缺少对应 Episode 候选重建器")
            if str(rebuilder._episode_rebuild_spec_sha256) != spec_sha256:
                raise ValueError("Checkpoint 上下文与 Episode 重建规格不匹配")
            context_refcounts[fingerprint] = context_refcounts.get(fingerprint, 0) + 1
            episode_refcounts[episode_id] = episode_refcounts.get(episode_id, 0) + 1
            return existing

        def intern_state_vector(
            fingerprint: str, values: np.ndarray | None
        ) -> np.ndarray:
            if values is None or not fingerprint:
                raise ValueError("Checkpoint Replay 状态特征或指纹缺失")
            state_vector = _state_vector(
                values, "checkpoint state_features", copy=False
            )
            existing = state_feature_pool.get(fingerprint)
            if existing is not None and not np.array_equal(existing, state_vector):
                raise ValueError("Checkpoint 相同状态指纹对应不同状态特征")
            if existing is None:
                existing = state_vector
                state_feature_pool[fingerprint] = existing
            state_feature_refcounts[fingerprint] = (
                state_feature_refcounts.get(fingerprint, 0) + 1
            )
            return existing

        rebuilt_items: list[ReplayTransition] = []
        for item in items:
            source_fingerprint = item.action.source_state_fingerprint
            current_state = intern_state_vector(
                source_fingerprint, item.state_features
            )
            next_state = (
                None
                if item.terminated
                else intern_state_vector(
                    str(item.next_state_fingerprint), item.next_state_features
                )
            )
            rebuilt_items.append(
                replace(
                    item,
                    state_features=current_state,
                    next_state_features=next_state,
                    current_candidate_rebuild_context_zlib=intern(
                        source_fingerprint,
                        item.current_candidate_rebuild_context_zlib,
                    ),
                    next_candidate_rebuild_context_zlib=(
                        None
                        if item.terminated
                        else intern(
                            str(item.next_state_fingerprint),
                            item.next_candidate_rebuild_context_zlib,
                        )
                    ),
                )
            )
        restored_rng = random.Random()
        restored_rng.setstate(state["rng_state"])
        if not 0 <= restored_position < self.capacity:
            raise ValueError("Checkpoint Replay Buffer 环形写入位置非法")
        if set(episode_refcounts) != set(candidate_rebuilders):
            raise ValueError("Checkpoint Replay 重建器与经验上下文集合不一致")
        if len(state_feature_pool) != int(
            state.get("unique_state_feature_count", -1)
        ):
            raise ValueError("Checkpoint Replay 唯一状态向量数量不一致")
        # 所有身份、上下文与随机状态校验通过后，才替换当前回放状态。
        self._items = rebuilt_items
        self._context_pool = context_pool
        self._context_refcounts = context_refcounts
        self._verified_candidate_payload_digests.clear()
        self._state_feature_pool = state_feature_pool
        self._state_feature_refcounts = state_feature_refcounts
        self._candidate_rebuilders = candidate_rebuilders
        self._episode_rebuild_refcounts = episode_refcounts
        self._position = restored_position
        self._rng = restored_rng
        self.write_count = restored_write_count
        self.sample_calls = restored_sample_calls
        self.sample_count = restored_sample_count


__all__ = ["ReplayTransition", "ReplayBuffer"]
