"""Tensor-parallel sharding utilities for HF-checkpoint loading.

Routed expert parameters are partitioned by *expert index*; everything
else is partitioned along a fixed dimension (``column-parallel`` /
``row-parallel`` / replicated) determined by the HF component name (see
:mod:`keys`).
"""

from __future__ import annotations

import re

import torch

from .keys import _get_shard_dim


def _expert_partition(
    n_experts: int, world_size: int, rank: int
) -> tuple[int, int]:
    """Return ``(start, end)`` expert indices owned by *rank*.

    Routed experts are partitioned by index: rank ``r`` owns experts
    ``[r * n_local, (r + 1) * n_local)``. Uneven divisibility would
    silently leave the last ``n_experts % world_size`` experts
    unassigned, so we fail fast instead. If a non-uniform layout is
    ever needed (e.g. heterogeneous GPUs), a divmod-based scheme can
    be added here once and consumed by both
    :func:`_is_foreign_expert` and :func:`_shard_tensor`.
    """
    if n_experts % world_size != 0:
        raise ValueError(
            f"n_experts ({n_experts}) is not divisible by world_size "
            f"({world_size}); even partitioning would leave "
            f"{n_experts % world_size} expert(s) unassigned. The "
            "sharded loader currently only supports uniform partitions."
        )
    n_local = n_experts // world_size
    start = rank * n_local
    return start, start + n_local


def _is_foreign_expert(
    hf_key: str, rank: int, world_size: int, n_experts: int
) -> bool:
    """Return True if this key is a routed expert that belongs to another rank."""
    expert_match = re.search(
        r"\.experts\.(\d+)\.", hf_key.replace("shared_experts", "_shared_")
    )
    if not expert_match:
        return False
    expert_idx = int(expert_match.group(1))
    start, end = _expert_partition(n_experts, world_size, rank)
    return not (start <= expert_idx < end)


def _shard_tensor(
    tensor: torch.Tensor,
    hf_key: str,
    rank: int,
    world_size: int,
    n_experts: int,
) -> torch.Tensor | None:
    """Shard a tensor for the given rank, or return it as-is if replicated.

    For routed expert parameters, returns the tensor only if the expert
    index belongs to this rank's partition; returns ``None`` otherwise.
    """
    if world_size <= 1:
        return tensor

    # Routed experts: partitioned by expert index.
    expert_match = re.search(
        r"\.experts\.(\d+)\.", hf_key.replace("shared_experts", "_shared_")
    )
    if expert_match:
        expert_idx = int(expert_match.group(1))
        start, end = _expert_partition(n_experts, world_size, rank)
        if start <= expert_idx < end:
            return tensor
        return None

    shard_dim = _get_shard_dim(hf_key)
    if shard_dim is None:
        return tensor

    dim_size = tensor.size(shard_dim)
    if dim_size % world_size != 0:
        raise ValueError(
            f"Cannot shard {hf_key} dim {shard_dim} (size {dim_size}) "
            f"across {world_size} ranks"
        )
    shard_size = dim_size // world_size
    return tensor.narrow(shard_dim, rank * shard_size, shard_size).contiguous()
