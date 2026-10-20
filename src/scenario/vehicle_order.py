"""Explicit vehicle ID mapping helpers for aligned scenario arrays."""

from __future__ import annotations

from collections.abc import Iterable


RESIDUAL_SAMPLING_CONTRACT = "GLOBAL_POOLED_WHOLE_BLOCK_V1"


def id_axis_indices(
    source_ids: Iterable[object],
    target_ids: Iterable[object],
    *,
    context: str,
) -> tuple[int, ...]:
    """Return source-axis indices in target ID order, requiring a one-to-one join."""

    source = tuple(str(value) for value in source_ids)
    target = tuple(str(value) for value in target_ids)
    for label, ids in (("source", source), ("target", target)):
        if any(not vehicle_id.strip() for vehicle_id in ids):
            raise ValueError(f"{context}: {label} vehicle IDs contain empty values")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{context}: {label} vehicle IDs contain duplicates")

    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        raise ValueError(
            f"{context}: vehicle ID sets do not align; "
            f"missing={missing}, unexpected={unexpected}"
        )

    source_index = {vehicle_id: index for index, vehicle_id in enumerate(source)}
    return tuple(source_index[vehicle_id] for vehicle_id in target)
