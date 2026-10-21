"""FlashSpeculativeMamba package exports."""

from src.backend import get_backend_info, print_backend_summary
from src.mamba_state_cache import (
    MambaStateSnapshot,
    snapshot_cache,
    restore_cache,
    create_mamba_cache,
    are_caches_equal,
    are_snapshots_equal,
)
from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.target_verifier import TargetVerifier, VerificationResult, apply_mamba_fast_multitoken_patch
from src.speculative_engine import (
    SpeculativeEngine,
    NaiveSpeculativeEngine,
    RoundRecord,
    SpeculativeGenerationResult,
)

__all__ = [
    "get_backend_info",
    "print_backend_summary",
    "MambaStateSnapshot",
    "snapshot_cache",
    "restore_cache",
    "create_mamba_cache",
    "are_caches_equal",
    "are_snapshots_equal",
    "DraftModel",
    "TargetModel",
    "TargetVerifier",
    "VerificationResult",
    "apply_mamba_fast_multitoken_patch",
    "SpeculativeEngine",
    "NaiveSpeculativeEngine",
    "RoundRecord",
    "SpeculativeGenerationResult",
]
