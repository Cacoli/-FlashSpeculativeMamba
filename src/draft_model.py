"""Draft model wrapper for autoregressive token drafting and recurrent state snapshotting.

Implements the draft model protocol specified in SPEC.md:
1. Prefill: Feeds prompt[:-1] to establish cache. The pending token p is prompt[-1].
2. Drafting (k steps):
   - Feeds p -> generates d1. Takes snapshot s0 (state after p).
   - Feeds d1 -> generates d2. Takes snapshot s1 (state after d1).
   - ...
   - Feeds d(k-1) -> generates dk. Takes snapshot s(k-1) (state after d(k-1)).
   - NOTE: dk is NEVER fed into the draft model during the drafting phase.
3. State update after target verification:
   - If n < k tokens accepted: restore snapshot s_n.
   - If n == k tokens accepted: restore s(k-1) and feed dk once to advance state.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from src.mamba_state_cache import (
    MambaStateSnapshot,
    snapshot_cache,
    restore_cache,
    create_mamba_cache,
)


class DraftModel:
    """Wraps state-spaces/mamba-130m-hf for speculative token proposal."""

    def __init__(
        self,
        model_id: str = "state-spaces/mamba-130m-hf",
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = MambaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device if self.device.type == "cuda" else None,
        )
        self.model.eval()

        self.cache: Optional[MambaCache] = None
        self.current_seq_len: int = 0

    def prefill(self, prompt_ids: torch.Tensor) -> Tuple[int, int]:
        """Prefill prompt[:-1] into draft cache.

        Args:
            prompt_ids: 2D LongTensor [1, sequence_length]

        Returns:
            Tuple of (pending_token_id, prefill_seq_len)
        """
        prompt_ids = prompt_ids.to(self.device)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(f"Expected prompt_ids shape [1, seq_len], got {prompt_ids.shape}")

        seq_len = prompt_ids.shape[1]
        if seq_len < 2:
            raise ValueError("Prompt must contain at least 2 tokens to have prefill + pending.")

        prefill_ids = prompt_ids[:, :-1]
        pending_token = int(prompt_ids[0, -1].item())
        self.current_seq_len = prefill_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(prefill_ids, use_cache=True)
        self.cache = outputs.cache_params

        return pending_token, self.current_seq_len

    def propose(
        self,
        pending_token: int,
        k: int,
    ) -> Tuple[List[int], List[MambaStateSnapshot]]:
        """Propose k draft tokens greedily with per-step state snapshotting.

        Args:
            pending_token: The pending token p from previous round.
            k: Number of draft tokens to propose.

        Returns:
            Tuple of (draft_tokens, snapshots):
            - draft_tokens: [d1, d2, ..., dk]
            - snapshots: [s0, s1, ..., s(k-1)] where s_j is the state after feeding candidate j.
        """
        if self.cache is None:
            raise RuntimeError("Must call prefill() before propose().")
        if k < 1:
            raise ValueError(f"Draft length k must be >= 1, got {k}")

        draft_tokens: List[int] = []
        snapshots: List[MambaStateSnapshot] = []

        curr_input = pending_token
        pos = self.current_seq_len

        with torch.no_grad():
            for step in range(k):
                input_t = torch.tensor([[curr_input]], device=self.device, dtype=torch.long)
                cache_pos = torch.tensor([pos], device=self.device, dtype=torch.long)

                out = self.model(
                    input_t,
                    cache_params=self.cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
                self.cache = out.cache_params
                pos += 1

                # Capture snapshot after feeding curr_input:
                # s0 is after feeding p; s1 is after feeding d1, ..., s(k-1) is after feeding d(k-1)
                snapshots.append(snapshot_cache(self.cache))

                next_tok = int(out.logits[0, -1].argmax().item())
                draft_tokens.append(next_tok)

                # Next input to feed
                curr_input = next_tok

        # Note: self.current_seq_len is not permanently updated here;
        # it is updated in update_state() after verification.
        return draft_tokens, snapshots

    def update_state(
        self,
        n_accepted: int,
        k: int,
        draft_tokens: List[int],
        snapshots: List[MambaStateSnapshot],
    ) -> None:
        """Roll back or advance draft cache state based on number of accepted tokens.

        Protocol from SPEC.md:
        - If n < k: restore snapshot s_n. The state is now after feeding accepted tokens.
        - If n == k: restore s(k-1) and feed dk once to advance state.
        """
        if self.cache is None:
            raise RuntimeError("Cache is uninitialized.")

        if n_accepted < k:
            # Restore state after feeding d_n (or p if n == 0)
            target_snapshot = snapshots[n_accepted]
            restore_cache(self.cache, target_snapshot)
            self.current_seq_len += n_accepted + 1
        else:
            # All k tokens accepted: restore s(k-1) and feed dk once
            restore_cache(self.cache, snapshots[k - 1])
            last_draft_token = draft_tokens[-1]
            pos = self.current_seq_len + k

            input_t = torch.tensor([[last_draft_token]], device=self.device, dtype=torch.long)
            cache_pos = torch.tensor([pos], device=self.device, dtype=torch.long)

            with torch.no_grad():
                out = self.model(
                    input_t,
                    cache_params=self.cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
            self.cache = out.cache_params
            self.current_seq_len += k + 1

    def generate_greedy(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> List[int]:
        """Greedy autoregressive generation baseline for draft model."""
        prompt_ids = prompt_ids.to(self.device)
        pending, _ = self.prefill(prompt_ids)
        generated = [pending]

        curr_token = pending
        pos = self.current_seq_len

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                input_t = torch.tensor([[curr_token]], device=self.device, dtype=torch.long)
                cache_pos = torch.tensor([pos], device=self.device, dtype=torch.long)

                out = self.model(
                    input_t,
                    cache_params=self.cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
                self.cache = out.cache_params
                pos += 1

                next_tok = int(out.logits[0, -1].argmax().item())
                generated.append(next_tok)
                curr_token = next_tok

                if curr_token == self.tokenizer.eos_token_id:
                    break

        return generated
