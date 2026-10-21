"""Speculative decoding engine for Mamba models.

Implements the speculative decoding protocol defined in SPEC.md:
1. Prefill: Feeds prompt[:-1] to initialize both draft and target models. The pending token p is prompt[-1].
2. For each round:
   a. Draft model proposes k tokens greedily: [d1, ..., dk], capturing per-step state snapshots.
   b. Target model verifies [p, d1, ..., dk] in one forward pass.
   c. Greedy accept d_j while argmax(logits[j-1]) == d_j. Let n = number accepted (0..k).
   d. Next pending token p' = argmax(logits[n]):
      - n == k: bonus token from target. Target state is already at end of round (no replay).
      - n < k: correction token. Target state rolls back to T0 and replays [p, d1..dn].
   e. Draft state update:
      - n < k: restores snapshot s_n.
      - n == k: restores snapshot s(k-1) and feeds dk once.
   f. Emits accepted tokens [d1..dn] and next pending token p'.
   g. Repeats until EOS or max_new_tokens reached.

Also implements NaiveSpeculativeEngine (the Stage 4 reference in SPEC.md) where target
recomputes the full context accumulated so far each round without persistent cache.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import torch
from transformers import AutoTokenizer

from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.target_verifier import VerificationResult


@dataclass
class RoundRecord:
    """Diagnostic details for a single speculative decoding round."""

    round_idx: int
    pending_in: int
    drafted_tokens: List[int]
    accepted_tokens: List[int]
    next_pending: int
    n_accepted: int
    k: int
    bonus_awarded: bool


@dataclass
class SpeculativeGenerationResult:
    """Metrics and generated token sequence from a speculative decoding run."""

    token_ids: List[int]
    text: str
    acceptance_rate: float
    mean_accepted_per_round: float
    total_drafted_tokens: int
    total_accepted_tokens: int
    num_rounds: int
    elapsed_time_sec: float
    tokens_per_sec: float
    round_history: List[RoundRecord] = field(default_factory=list)


class SpeculativeEngine:
    """Cached Mamba speculative decoding engine."""

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        k: int = 4,
        use_fast_multitoken: bool = True,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
        self.use_fast_multitoken = use_fast_multitoken
        self.tokenizer = target_model.tokenizer

        # Configure target verifier mode
        self.target_model.verifier.use_fast_multitoken = self.use_fast_multitoken

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> SpeculativeGenerationResult:
        """Run cached speculative decoding until max_new_tokens or EOS.

        Args:
            prompt_ids: 2D LongTensor [1, sequence_length]
            max_new_tokens: Maximum new tokens to emit (including first pending token).
            k: Optional override for draft length (defaults to self.k).
            eos_token_id: Optional token ID to terminate generation (defaults to tokenizer.eos_token_id).

        Returns:
            SpeculativeGenerationResult containing tokens, speed, and acceptance metrics.
        """
        active_k = k if k is not None else self.k
        if active_k < 1:
            raise ValueError(f"k must be >= 1, got {active_k}")

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Step 1: Prefill both models
        p_draft, len_draft = self.draft_model.prefill(prompt_ids)
        p_target, len_target = self.target_model.prefill(prompt_ids)
        assert p_draft == p_target, f"Prefill pending mismatch: {p_draft} vs {p_target}"

        pending_token = p_target
        emitted_tokens: List[int] = [pending_token]

        total_drafted = 0
        total_accepted = 0
        rounds: List[RoundRecord] = []
        round_idx = 0

        # Step 2: Speculative decoding loop
        while len(emitted_tokens) < max_new_tokens:
            if pending_token == eos_token_id:
                break

            # Bound draft length by remaining token budget
            remaining = max_new_tokens - len(emitted_tokens)
            current_k = min(active_k, remaining)
            if current_k < 1:
                break

            # Draft proposes current_k tokens greedily
            draft_tokens, snapshots = self.draft_model.propose(pending_token, k=current_k)
            total_drafted += len(draft_tokens)

            # Target verifies candidate sequence [p, d1..dk]
            verif_res: VerificationResult = self.target_model.verify_draft(
                pending_token, draft_tokens
            )
            total_accepted += verif_res.n_accepted

            # Synchronize draft model recurrent state
            self.draft_model.update_state(
                verif_res.n_accepted, current_k, draft_tokens, snapshots
            )

            # Record round
            rounds.append(
                RoundRecord(
                    round_idx=round_idx,
                    pending_in=pending_token,
                    drafted_tokens=draft_tokens,
                    accepted_tokens=verif_res.accepted_tokens,
                    next_pending=verif_res.next_pending_token,
                    n_accepted=verif_res.n_accepted,
                    k=current_k,
                    bonus_awarded=verif_res.bonus_token_awarded,
                )
            )
            round_idx += 1

            # Emit accepted tokens
            eos_hit = False
            for tok in verif_res.accepted_tokens:
                emitted_tokens.append(tok)
                if tok == eos_token_id or len(emitted_tokens) >= max_new_tokens:
                    eos_hit = True
                    break

            if eos_hit:
                break

            # Emit next pending token (bonus or correction)
            pending_token = verif_res.next_pending_token
            emitted_tokens.append(pending_token)
            if pending_token == eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0
        mean_acc = (total_accepted / len(rounds)) if rounds else 0.0
        tps = len(emitted_tokens) / elapsed if elapsed > 0 else 0.0

        decoded_text = self.tokenizer.decode(emitted_tokens, skip_special_tokens=False)

        return SpeculativeGenerationResult(
            token_ids=emitted_tokens,
            text=decoded_text,
            acceptance_rate=acc_rate,
            mean_accepted_per_round=mean_acc,
            total_drafted_tokens=total_drafted,
            total_accepted_tokens=total_accepted,
            num_rounds=len(rounds),
            elapsed_time_sec=elapsed,
            tokens_per_sec=tps,
            round_history=rounds,
        )


class NaiveSpeculativeEngine:
    """Naive reference speculative decoding engine (Stage 4 reference in SPEC.md).

    The target model recomputes the full context accumulated so far on each round
    without using persistent recurrent cache. The cached engine must match this.
    """

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        k: int = 4,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
        self.tokenizer = target_model.tokenizer
        self.device = target_model.device

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> SpeculativeGenerationResult:
        """Run naive speculative decoding recomputing full context each round."""
        active_k = k if k is not None else self.k
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        prompt_ids = prompt_ids.to(self.device)
        full_tokens = prompt_ids.clone()  # Accumulates prompt + emitted tokens
        pending_token = int(prompt_ids[0, -1].item())
        emitted_tokens: List[int] = [pending_token]

        # Prefill draft model
        self.draft_model.prefill(prompt_ids)

        total_drafted = 0
        total_accepted = 0
        rounds: List[RoundRecord] = []
        round_idx = 0

        while len(emitted_tokens) < max_new_tokens:
            if pending_token == eos_token_id:
                break

            remaining = max_new_tokens - len(emitted_tokens)
            current_k = min(active_k, remaining)
            if current_k < 1:
                break

            # Draft model proposes current_k tokens
            draft_tokens, snapshots = self.draft_model.propose(pending_token, k=current_k)
            total_drafted += len(draft_tokens)

            # Target model verifies candidates by feeding full_context + draft_tokens in ONE forward pass
            cand_tensor = torch.tensor([draft_tokens], device=self.device, dtype=torch.long)
            full_round_input = torch.cat([full_tokens, cand_tensor], dim=1)

            # Full forward pass without cache
            with torch.no_grad():
                out = self.target_model.model(full_round_input, use_cache=False)
            logits = out.logits  # [1, full_len + k, vocab_size]

            # The logits corresponding to [full_tokens[-1], d1, ..., d(k-1)]:
            # Token at full_tokens.shape[1] - 1 predicts draft_tokens[0]
            start_pos = full_tokens.shape[1] - 1
            verif_logits = logits[0, start_pos : start_pos + current_k + 1]

            n_accepted = 0
            accepted_tokens: List[int] = []
            for j in range(current_k):
                pred = int(verif_logits[j].argmax().item())
                if pred == draft_tokens[j]:
                    n_accepted += 1
                    accepted_tokens.append(draft_tokens[j])
                else:
                    break

            next_pending = int(verif_logits[n_accepted].argmax().item())
            bonus_awarded = n_accepted == current_k
            total_accepted += n_accepted

            # Synchronize draft model
            self.draft_model.update_state(n_accepted, current_k, draft_tokens, snapshots)

            rounds.append(
                RoundRecord(
                    round_idx=round_idx,
                    pending_in=pending_token,
                    drafted_tokens=draft_tokens,
                    accepted_tokens=accepted_tokens,
                    next_pending=next_pending,
                    n_accepted=n_accepted,
                    k=current_k,
                    bonus_awarded=bonus_awarded,
                )
            )
            round_idx += 1

            # Update full_tokens and emitted_tokens
            eos_hit = False
            for tok in accepted_tokens:
                emitted_tokens.append(tok)
                new_t = torch.tensor([[tok]], device=self.device, dtype=torch.long)
                full_tokens = torch.cat([full_tokens, new_t], dim=1)
                if tok == eos_token_id or len(emitted_tokens) >= max_new_tokens:
                    eos_hit = True
                    break

            if eos_hit:
                break

            pending_token = next_pending
            emitted_tokens.append(pending_token)
            new_p = torch.tensor([[pending_token]], device=self.device, dtype=torch.long)
            full_tokens = torch.cat([full_tokens, new_p], dim=1)
            if pending_token == eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0
        mean_acc = (total_accepted / len(rounds)) if rounds else 0.0
        tps = len(emitted_tokens) / elapsed if elapsed > 0 else 0.0

        decoded_text = self.tokenizer.decode(emitted_tokens, skip_special_tokens=False)

        return SpeculativeGenerationResult(
            token_ids=emitted_tokens,
            text=decoded_text,
            acceptance_rate=acc_rate,
            mean_accepted_per_round=mean_acc,
            total_drafted_tokens=total_drafted,
            total_accepted_tokens=total_accepted,
            num_rounds=len(rounds),
            elapsed_time_sec=elapsed,
            tokens_per_sec=tps,
            round_history=rounds,
        )
