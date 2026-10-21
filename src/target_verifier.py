"""Target verifier for Mamba speculative decoding.

Implements candidate verification according to SPEC.md:
Given pending token p and draft tokens [d1, ..., dk]:
1. Snapshot target state T0.
2. Feed [p, d1, ..., dk] forward to obtain logits for each position 0..k.
   - Fast path: Patched multi-token forward pass with exact causal convolution history.
   - Sequential fallback: Token-by-token feed.
3. Accept d_j while argmax(logits[j-1]) == d_j. Let n = number accepted (0..k).
4. Next pending token p' = argmax(logits[n]):
   - n == k: bonus token from target model. Target state is already at end of round.
   - n < k: correction token from target model. Restore T0 and replay [p, d1..dn] to correct state.
5. Returns accepted tokens, next pending token, and acceptance count.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import MambaForCausalLM
from transformers.cache_utils import MambaCache
from transformers.models.mamba.modeling_mamba import MambaMixer

from src.mamba_state_cache import snapshot_cache, restore_cache, MambaStateSnapshot


@dataclass
class VerificationResult:
    """Outcome of verifying a draft sequence against the target model."""

    accepted_tokens: List[int]
    next_pending_token: int
    n_accepted: int
    total_drafted: int
    bonus_token_awarded: bool


_PATCH_APPLIED = False


def apply_mamba_fast_multitoken_patch() -> None:
    """Patch MambaMixer.slow_forward to enable multi-token forward passes from a non-empty cache.

    In stock transformers 4.47.1, slow_forward assumes seq_len == 1 when cache_params is present,
    causing IndexError during multi-token verification. This patch computes exact causal convolution
    over the concatenated historical buffer without spurious padding, matching sequential feed.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    orig_slow_forward = MambaMixer.slow_forward

    def patched_slow_forward(
        self,
        input_states: torch.Tensor,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone().to(hidden_states.device)
            if (
                cache_position is not None
                and cache_position.shape[0] == self.conv_kernel_size
                and seq_len >= self.conv_kernel_size
            ):
                # Prefill branch
                conv_state = F.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
            elif seq_len == 1:
                # Single-token incremental decode branch
                conv_state = cache_params.update_conv_state(
                    self.layer_idx, hidden_states, cache_position
                )
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                # Multi-token verification branch:
                # Prepend the last (conv_kernel_size - 1) historical tokens from cache conv_states
                past_conv = cache_params.conv_states[self.layer_idx][
                    :, :, -(self.conv_kernel_size - 1) :
                ]
                conv_input = torch.cat([past_conv, hidden_states], dim=-1)
                # Convolve with padding=0 to avoid extra left padding
                conv_out = F.conv1d(
                    conv_input,
                    self.conv1d.weight,
                    bias=self.conv1d.bias if self.use_conv_bias else None,
                    padding=0,
                    groups=self.intermediate_size,
                )
                hidden_states = self.act(conv_out).to(dtype)
                # Update cache conv_states with the last conv_kernel_size tokens
                cache_params.conv_states[self.layer_idx].copy_(
                    conv_input[:, :, -self.conv_kernel_size :]
                )
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device,
                dtype=dtype,
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # SSM sequence transformation
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = F.softplus(discrete_time_step).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    MambaMixer.slow_forward = patched_slow_forward
    _PATCH_APPLIED = True


class TargetVerifier:
    """Verifies candidate draft tokens against a target Mamba model."""

    def __init__(
        self,
        model: MambaForCausalLM,
        use_fast_multitoken: bool = True,
    ) -> None:
        self.model = model
        self.device = model.device
        self.use_fast_multitoken = use_fast_multitoken

        if self.use_fast_multitoken:
            apply_mamba_fast_multitoken_patch()

    def verify_candidate(
        self,
        pending_token: int,
        draft_tokens: List[int],
        target_cache: MambaCache,
        current_seq_len: int,
    ) -> VerificationResult:
        """Verify proposed draft tokens against target model logits.

        Args:
            pending_token: The last emitted token fed at the start of the round.
            draft_tokens: List of k token IDs proposed by the draft model.
            target_cache: Active MambaCache of the target model (mutated in-place to correct end-of-round state).
            current_seq_len: Total number of tokens processed into target_cache so far.

        Returns:
            VerificationResult detailing accepted tokens, correction/bonus token, and count.
        """
        k = len(draft_tokens)
        if k == 0:
            raise ValueError("Draft tokens list must contain at least 1 token.")

        candidate_ids = [pending_token] + draft_tokens
        cand_tensor = torch.tensor(
            [candidate_ids], device=self.device, dtype=torch.long
        )  # [1, k+1]

        # Snapshot T0 before verification forward pass
        snapshot_t0 = snapshot_cache(target_cache)

        if self.use_fast_multitoken:
            cache_pos = torch.arange(
                current_seq_len, current_seq_len + k + 1, device=self.device
            )
            with torch.no_grad():
                outputs = self.model(
                    cand_tensor,
                    cache_params=target_cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
            logits = outputs.logits  # [1, k+1, vocab_size]
        else:
            # Sequential fallback verification
            logits_list = []
            pos = current_seq_len
            with torch.no_grad():
                for i in range(k + 1):
                    tok = cand_tensor[:, i : i + 1]
                    out_step = self.model(
                        tok,
                        cache_params=target_cache,
                        cache_position=torch.tensor([pos], device=self.device),
                        use_cache=True,
                    )
                    logits_list.append(out_step.logits)
                    pos += 1
            logits = torch.cat(logits_list, dim=1)  # [1, k+1, vocab_size]

        # Determine acceptance
        # logits[0, j] is the target model's prediction for candidate_ids[j+1]
        # candidate_ids[0] = p -> logits[0] predicts d1 (draft_tokens[0])
        # candidate_ids[1] = d1 -> logits[1] predicts d2 (draft_tokens[1])
        # candidate_ids[k] = dk -> logits[k] predicts bonus token p'
        n_accepted = 0
        accepted_tokens: List[int] = []

        for j in range(k):
            pred_token = int(logits[0, j].argmax().item())
            expected_draft = draft_tokens[j]
            if pred_token == expected_draft:
                n_accepted += 1
                accepted_tokens.append(expected_draft)
            else:
                # First mismatch encountered
                break

        # Next pending token p'
        # If n == k: p' = argmax(logits[k]) (bonus token)
        # If n < k: p' = argmax(logits[n]) (correction token replacing draft_tokens[n])
        next_pending_token = int(logits[0, n_accepted].argmax().item())
        bonus_token_awarded = n_accepted == k

        # State management per SPEC.md:
        # - n == k: Target state is already at the end of [p, d1..dk], perfectly up to date!
        # - n < k: Target state in target_cache advanced through all k+1 tokens. We must restore T0
        #          and replay [p, d1..dn] so target state is exactly up to date before p'.
        if not bonus_token_awarded:
            restore_cache(target_cache, snapshot_t0)
            if n_accepted + 1 > 0:
                replay_ids = candidate_ids[: n_accepted + 1]  # [p, d1, ..., dn]
                replay_tensor = torch.tensor(
                    [replay_ids], device=self.device, dtype=torch.long
                )
                replay_pos = torch.arange(
                    current_seq_len,
                    current_seq_len + len(replay_ids),
                    device=self.device,
                )
                with torch.no_grad():
                    self.model(
                        replay_tensor,
                        cache_params=target_cache,
                        cache_position=replay_pos,
                        use_cache=True,
                    )

        return VerificationResult(
            accepted_tokens=accepted_tokens,
            next_pending_token=next_pending_token,
            n_accepted=n_accepted,
            total_drafted=k,
            bonus_token_awarded=bonus_token_awarded,
        )
