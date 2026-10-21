"""Target model wrapper for greedy autoregressive generation and speculative verification.

Wraps state-spaces/mamba-1.4b-hf (with fallback to mamba-790m-hf) in strict fp16 precision.
Provides:
1. Prefill and cached greedy autoregressive generation (the target-only performance baseline).
2. Naive full-context recompute generation (the Stage 4 correctness reference).
3. Verification integration via TargetVerifier.
"""

from __future__ import annotations
import gc
from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from src.mamba_state_cache import snapshot_cache, restore_cache
from src.target_verifier import TargetVerifier, VerificationResult


class TargetModel:
    """Wraps state-spaces/mamba-1.4b-hf for greedy baseline generation and verification."""

    def __init__(
        self,
        model_id: str = "state-spaces/mamba-1.4b-hf",
        fallback_model_id: str = "state-spaces/mamba-790m-hf",
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float16,
        use_fast_multitoken: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.active_model_id = model_id
        self.use_fast_multitoken = use_fast_multitoken

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Attempt loading primary target model (1.4B in fp16 takes ~2.8 GB VRAM)
        try:
            print(f"Loading target model '{model_id}' in float16 on {self.device}...")
            self.model = MambaForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device if self.device.type == "cuda" else None,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and fallback_model_id:
                print(f"[WARNING] VRAM exhausted while loading {model_id}.")
                print(f"          Falling back to {fallback_model_id} per SPEC.md...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.active_model_id = fallback_model_id
                self.model = MambaForCausalLM.from_pretrained(
                    fallback_model_id,
                    torch_dtype=self.dtype,
                    device_map=self.device if self.device.type == "cuda" else None,
                )
            else:
                raise

        self.model.eval()
        self.cache: Optional[MambaCache] = None
        self.current_seq_len: int = 0
        self.verifier = TargetVerifier(self.model, use_fast_multitoken=self.use_fast_multitoken)

    def prefill(self, prompt_ids: torch.Tensor) -> Tuple[int, int]:
        """Prefill prompt[:-1] into target cache.

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

    def generate_greedy(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Cached greedy autoregressive generation (the target-only performance baseline).

        Generates max_new_tokens tokens using recurrent state caching.
        """
        prompt_ids = prompt_ids.to(self.device)
        pending, _ = self.prefill(prompt_ids)
        generated = [pending]

        curr_token = pending
        pos = self.current_seq_len
        target_eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id

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

                if target_eos is not None and target_eos >= 0 and curr_token == target_eos:
                    break

        return generated

    def generate_naive_reference(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Naive full-context recompute reference generation (Stage 4 reference).

        At every step, feeds the entire token sequence accumulated so far without any
        persistent cache. Serves as the ground-truth reference for exact greedy equivalence.
        """
        prompt_ids = prompt_ids.to(self.device)
        full_ids = prompt_ids.clone()
        pending = int(prompt_ids[0, -1].item())
        generated = [pending]
        target_eos = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                out = self.model(full_ids, use_cache=False)
                next_tok = int(out.logits[0, -1].argmax().item())
                generated.append(next_tok)

                if target_eos is not None and target_eos >= 0 and next_tok == target_eos:
                    break

                next_tensor = torch.tensor([[next_tok]], device=self.device, dtype=torch.long)
                full_ids = torch.cat([full_ids, next_tensor], dim=1)

        return generated

    def verify_draft(
        self,
        pending_token: int,
        draft_tokens: List[int],
    ) -> VerificationResult:
        """Verify proposed draft tokens using the active target cache."""
        if self.cache is None:
            raise RuntimeError("Must call prefill() before verify_draft().")

        result = self.verifier.verify_candidate(
            pending_token=pending_token,
            draft_tokens=draft_tokens,
            target_cache=self.cache,
            current_seq_len=self.current_seq_len,
        )

        # Update target sequence length
        if result.bonus_token_awarded:
            self.current_seq_len += result.total_drafted + 1
        else:
            self.current_seq_len += result.n_accepted + 1

        return result
