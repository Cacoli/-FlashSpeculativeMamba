"""Empirical investigation script for the Known Risk defined in SPEC.md:

'The verify step assumes one multi-token forward pass from a non-empty cache gives
 the same logits and final state as feeding tokens one at a time. Test this on
 the installed transformers version (fp16). If it fails, fall back to sequential
 verification and report it.'
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

# Ensure src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mamba_state_cache import snapshot_cache, restore_cache, are_caches_equal


def run_empirical_risk_test(model_id: str = "state-spaces/mamba-130m-hf") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} in float16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MambaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    model.eval()

    prompt = "The quick brown fox jumps over the lazy dog and discovers"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Split into prefill (all tokens except last) and pending token (last prompt token)
    prefill_ids = prompt_ids[:, :-1]
    pending_token = prompt_ids[:, -1:]
    seq_len_prefill = prefill_ids.shape[1]

    print(f"Prefill tokens count: {seq_len_prefill}, pending token: {pending_token.item()}")

    # 1. Run Prefill to establish non-empty MambaCache
    with torch.no_grad():
        prefill_out = model(prefill_ids, use_cache=True)
    base_cache: MambaCache = prefill_out.cache_params
    base_snapshot = snapshot_cache(base_cache)

    print("Initial prefill complete. Base cache snapshot taken.")

    # Synthetic draft tokens (k=4)
    draft_tokens = torch.tensor([[100, 250, 420, 1024]], device=device, dtype=torch.long)
    full_candidate = torch.cat([pending_token, draft_tokens], dim=1) # Shape: [1, 5]
    k_plus_1 = full_candidate.shape[1]

    # --- Strategy A: Sequential feeding token-by-token ---
    print("\n--- Strategy A: Sequential single-token forward passes ---")
    cache_a = MambaCache(model.config, batch_size=1, device=device, dtype=torch.float16)
    restore_cache(cache_a, base_snapshot)

    seq_logits_list = []
    # Feed tokens one by one
    curr_pos = seq_len_prefill
    with torch.no_grad():
        for i in range(k_plus_1):
            tok = full_candidate[:, i : i + 1]
            cache_pos = torch.tensor([curr_pos], device=device, dtype=torch.long)
            out_step = model(tok, cache_params=cache_a, cache_position=cache_pos, use_cache=True)
            seq_logits_list.append(out_step.logits) # [1, 1, vocab]
            curr_pos += 1
            cache_a = out_step.cache_params

    seq_logits = torch.cat(seq_logits_list, dim=1) # [1, 5, vocab]
    print(f"Sequential forward completed. Output logits shape: {seq_logits.shape}")

    # --- Strategy B: Single multi-token forward pass ---
    print("\n--- Strategy B: One multi-token forward pass [p, d1..dk] ---")
    cache_b = MambaCache(model.config, batch_size=1, device=device, dtype=torch.float16)
    restore_cache(cache_b, base_snapshot)

    multi_token_succeeded = False
    error_message = None
    multi_logits = None
    logits_max_diff = None
    caches_match = False

    try:
        cache_pos_multi = torch.arange(seq_len_prefill, seq_len_prefill + k_plus_1, device=device, dtype=torch.long)
        with torch.no_grad():
            out_multi = model(
                full_candidate,
                cache_params=cache_b,
                cache_position=cache_pos_multi,
                use_cache=True,
            )
        multi_logits = out_multi.logits
        multi_token_succeeded = True
        print(f"Multi-token forward executed without crash! Output logits shape: {multi_logits.shape}")
        
        # Check shapes
        if multi_logits.shape != seq_logits.shape:
            print(f"[FAIL] Shape mismatch: multi={multi_logits.shape} vs seq={seq_logits.shape}")
            diff_str = f"Shape mismatch: {multi_logits.shape} vs {seq_logits.shape}"
        else:
            diff = (multi_logits - seq_logits).abs()
            logits_max_diff = diff.max().item()
            print(f"Max absolute logit difference: {logits_max_diff:.6f}")
            caches_match = are_caches_equal(cache_a, cache_b)
            print(f"Final MambaCache exactly equal: {caches_match}")
    except Exception as e:
        error_message = str(e)
        print(f"[FAIL] Multi-token forward pass raised exception:\n  {type(e).__name__}: {e}")

    # Fallback assessment
    requires_fallback = (not multi_token_succeeded) or (logits_max_diff is not None and logits_max_diff > 1e-2) or (not caches_match)

    report = {
        "multi_token_succeeded": multi_token_succeeded,
        "error_message": error_message,
        "logits_max_diff": logits_max_diff,
        "caches_match": caches_match,
        "requires_sequential_fallback": requires_fallback,
    }

    print("\n" + "=" * 60)
    print("EMPIRICAL RISK ASSESSMENT SUMMARY")
    print("=" * 60)
    print(f"Multi-token forward allowed by HF    : {multi_token_succeeded}")
    if error_message:
        print(f"Error Encountered                   : {error_message}")
    print(f"Max Logit Difference                : {logits_max_diff}")
    print(f"Cache Recurrent States Equal        : {caches_match}")
    print(f"Decision for target_verifier        : {'FALL BACK TO SEQUENTIAL VERIFICATION' if requires_fallback else 'USE PARALLEL MULTI-TOKEN VERIFICATION'}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_empirical_risk_test()
