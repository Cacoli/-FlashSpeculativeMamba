"""Mamba state cache management, snapshotting, and rollback for speculative decoding.

Mamba state per layer consists of two fixed-size buffers:
1. `conv_states`: Shape [num_layers, batch_size, intermediate_size, conv_kernel]
   Holds the 1D causal convolution buffer across historical sequence steps.
2. `ssm_states`: Shape [num_layers, batch_size, intermediate_size, ssm_state_size]
   Holds the continuous state-space recurrent hidden state h_t.

Unlike Transformer KV-caches which grow with sequence length and can be sliced to roll back,
Mamba states are fixed-size and cannot simply be sliced. Rollback is achieved by
saving a snapshot (clone) of the tensors and restoring them in-place via `.copy_()`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
from transformers.cache_utils import MambaCache
from transformers.models.mamba.configuration_mamba import MambaConfig


@dataclass
class MambaStateSnapshot:
    """Immutable snapshot of a MambaCache's internal recurrent state at a specific step."""

    conv_states: torch.Tensor
    ssm_states: torch.Tensor

    def restore_into(self, cache: MambaCache) -> None:
        """In-place restore snapshot tensors into an existing active MambaCache.

        Uses in-place .copy_() to avoid reallocating cache buffers or breaking Dynamo
        static address markings.
        """
        if cache.conv_states.shape != self.conv_states.shape:
            raise ValueError(
                f"Shape mismatch in conv_states during restore: active {cache.conv_states.shape} vs snapshot {self.conv_states.shape}"
            )
        if cache.ssm_states.shape != self.ssm_states.shape:
            raise ValueError(
                f"Shape mismatch in ssm_states during restore: active {cache.ssm_states.shape} vs snapshot {self.ssm_states.shape}"
            )

        cache.conv_states.copy_(self.conv_states)
        cache.ssm_states.copy_(self.ssm_states)

    def clone(self) -> MambaStateSnapshot:
        """Create a deep copy of this state snapshot."""
        return MambaStateSnapshot(
            conv_states=self.conv_states.clone(),
            ssm_states=self.ssm_states.clone(),
        )

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> MambaStateSnapshot:
        """Move snapshot tensors to another device or precision."""
        return MambaStateSnapshot(
            conv_states=self.conv_states.to(device=device, dtype=dtype),
            ssm_states=self.ssm_states.to(device=device, dtype=dtype),
        )

    @property
    def byte_size(self) -> int:
        """Total memory consumption of the snapshot tensors in bytes."""
        return (
            self.conv_states.numel() * self.conv_states.element_size()
            + self.ssm_states.numel() * self.ssm_states.element_size()
        )


def snapshot_cache(cache: MambaCache) -> MambaStateSnapshot:
    """Capture a deep-copy snapshot of an active MambaCache's internal recurrent state.

    Args:
        cache: The active MambaCache instance to snapshot.

    Returns:
        MambaStateSnapshot containing cloned conv_states and ssm_states tensors.
    """
    return MambaStateSnapshot(
        conv_states=cache.conv_states.clone(),
        ssm_states=cache.ssm_states.clone(),
    )


def restore_cache(cache: MambaCache, snapshot: MambaStateSnapshot) -> None:
    """Restore saved snapshot tensors back into an active MambaCache in-place.

    Args:
        cache: The active MambaCache instance to mutate.
        snapshot: The MambaStateSnapshot to restore from.
    """
    snapshot.restore_into(cache)


def create_mamba_cache(
    config: MambaConfig,
    batch_size: int = 1,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float16,
) -> MambaCache:
    """Instantiate a clean, zeroed MambaCache for a given MambaConfig.

    Args:
        config: MambaConfig describing layer count, intermediate_size, conv_kernel, etc.
        batch_size: Batch size for cache allocation (default 1).
        device: Target torch device (e.g. 'cuda' or 'cpu').
        dtype: Numerical precision (fp16 strictly per SPEC.md).

    Returns:
        A freshly allocated MambaCache with zeroed recurrent states.
    """
    return MambaCache(
        config=config,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )


def are_caches_equal(cache_a: MambaCache, cache_b: MambaCache) -> bool:
    """Check whether two MambaCaches have exactly identical conv_states and ssm_states."""
    return torch.equal(cache_a.conv_states, cache_b.conv_states) and torch.equal(
        cache_a.ssm_states, cache_b.ssm_states
    )


def are_snapshots_equal(snap_a: MambaStateSnapshot, snap_b: MambaStateSnapshot) -> bool:
    """Check whether two MambaStateSnapshots have exactly identical states."""
    return torch.equal(snap_a.conv_states, snap_b.conv_states) and torch.equal(
        snap_a.ssm_states, snap_b.ssm_states
    )
