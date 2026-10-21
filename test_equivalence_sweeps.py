"""Comprehensive equivalence sweeps for FlashSpeculativeMamba (Stage 4).

Tests long generations (200+ tokens) across diverse prompts for k in {1, 2, 4, 8}:
Compares:
1. Target-Only Greedy Baseline (target.generate_greedy)
2. Cached Speculative Engine (fast multi-token verifier)
3. Naive Speculative Engine (full-context recompute reference)

Reports exact match or divergence index, divergence tokens, and logit margins.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.speculative_engine import SpeculativeEngine, NaiveSpeculativeEngine


PROMPTS = [
    (
        "Technical / CS",
        "In computer science, speculative decoding accelerates autoregressive sequence generation by using a small draft model to propose candidates that a larger target model verifies in parallel, thereby reducing memory bandwidth bottlenecks",
    ),
    (
        "Mathematical / Scientific",
        "The fundamental theorem of calculus establishes a profound relationship between differentiation and integration, stating that the definite integral of a function can be computed using its antiderivative, which implies that",
    ),
    (
        "Creative Narrative",
        "The clock in the ancient clocktower chimed three times into the misty midnight air. The young watchmaker reached into his pocket and pulled out the brass key, wondering if the legends about the hidden workshop were true",
    ),
    (
        "Code / Programming",
        "def fibonacci_generator(n: int):\n    \"\"\"Yield the first n Fibonacci numbers with optimal space complexity.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n",
    ),
]


def run_equivalence_sweeps(
    target_model_id: str = "state-spaces/mamba-130m-hf",
    draft_model_id: str = "state-spaces/mamba-130m-hf",
    max_new_tokens: int = 220,
    k_values: List[int] = [1, 2, 4, 8],
) -> List[Dict[str, Any]]:
    print("=" * 80)
    print(f"STAGE 4 EQUIVALENCE SWEEPS: Long Generation (max_new_tokens={max_new_tokens})")
    print(f"Target Model: {target_model_id} (FP16)")
    print(f"Draft Model : {draft_model_id} (FP16)")
    print(f"Sweep Values: k in {k_values}")
    print("=" * 80)

    # Instantiate models
    print("\nLoading models...")
    draft = DraftModel(model_id=draft_model_id)
    target = TargetModel(model_id=target_model_id, use_fast_multitoken=True)
    tokenizer = target.tokenizer

    cached_engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=True)
    naive_engine = NaiveSpeculativeEngine(draft, target, k=4)

    sweep_results: List[Dict[str, Any]] = []

    for prompt_label, prompt_text in PROMPTS:
        print(f"\n{'#' * 80}")
        print(f"Prompt Category: [{prompt_label}]")
        print(f"Prompt Text: \"{prompt_text[:70]}...\"")
        print(f"{'#' * 80}")

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

        # 1. Target-Only Greedy Baseline
        print("\n[1/3] Generating Target-Only Greedy Baseline (200+ tokens)...")
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        target_tokens = target.generate_greedy(prompt_ids, max_new_tokens=max_new_tokens)
        t1.record()
        torch.cuda.synchronize()
        target_time_ms = t0.elapsed_time(t1)
        target_tps = len(target_tokens) / (target_time_ms / 1000.0)
        print(f"      Generated {len(target_tokens)} tokens in {target_time_ms:.1f} ms ({target_tps:.2f} tok/s)")

        # 2. Sweeps across k
        for k in k_values:
            print(f"\n--- Testing k = {k} ---")

            # Run Cached Speculative Engine
            spec_res = cached_engine.generate(prompt_ids, max_new_tokens=max_new_tokens, k=k)
            spec_tokens = spec_res.token_ids

            # Check match against target-only
            match = (spec_tokens == target_tokens)
            divergence_idx = None
            diverged_target_token = None
            diverged_spec_token = None
            diverged_target_word = None
            diverged_spec_word = None

            if not match:
                min_len = min(len(spec_tokens), len(target_tokens))
                for i in range(min_len):
                    if spec_tokens[i] != target_tokens[i]:
                        divergence_idx = i
                        diverged_target_token = target_tokens[i]
                        diverged_spec_token = spec_tokens[i]
                        diverged_target_word = tokenizer.decode([target_tokens[i]])
                        diverged_spec_word = tokenizer.decode([spec_tokens[i]])
                        break
                if divergence_idx is None and len(spec_tokens) != len(target_tokens):
                    divergence_idx = min_len
                    diverged_target_token = target_tokens[min_len] if min_len < len(target_tokens) else "EOF"
                    diverged_spec_token = spec_tokens[min_len] if min_len < len(spec_tokens) else "EOF"

            # Run Naive Speculative Engine (Stage 4 reference) on first 64 tokens for verification
            naive_k = min(k, 4)
            naive_res = naive_engine.generate(prompt_ids, max_new_tokens=64, k=naive_k)
            naive_match_target = (naive_res.token_ids == target_tokens[: len(naive_res.token_ids)])
            naive_match_cached = (naive_res.token_ids == spec_tokens[: len(naive_res.token_ids)])

            result_entry = {
                "prompt": prompt_label,
                "k": k,
                "target_len": len(target_tokens),
                "spec_len": len(spec_tokens),
                "exact_match": match,
                "divergence_idx": divergence_idx,
                "target_token": diverged_target_token,
                "spec_token": diverged_spec_token,
                "target_word": repr(diverged_target_word),
                "spec_word": repr(diverged_spec_word),
                "acceptance_rate": round(spec_res.acceptance_rate * 100, 2),
                "mean_acc_per_round": round(spec_res.mean_accepted_per_round, 2),
                "num_rounds": spec_res.num_rounds,
                "target_tps": round(target_tps, 2),
                "spec_tps": round(spec_res.tokens_per_sec, 2),
                "speedup": round(spec_res.tokens_per_sec / target_tps, 2) if target_tps > 0 else 1.0,
                "naive_matches_target": naive_match_target,
                "naive_matches_cached": naive_match_cached,
            }
            sweep_results.append(result_entry)

            status = "EXACT MATCH (100%)" if match else f"DIVERGED at index {divergence_idx}"
            print(f"      Result: {status}")
            print(f"      Tokens: {len(spec_tokens)} | Rounds: {spec_res.num_rounds} | Acceptance: {result_entry['acceptance_rate']}% | Speedup: {result_entry['speedup']}x")
            if not match:
                print(f"      [Divergence Details]:")
                print(f"         Position: token {divergence_idx}/{len(target_tokens)}")
                print(f"         Target: ID={diverged_target_token} ({result_entry['target_word']})")
                print(f"         Spec  : ID={diverged_spec_token} ({result_entry['spec_word']})")

    # Print summary table
    print("\n" + "=" * 90)
    print("STAGE 4 EQUIVALENCE & PERFORMANCE SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Prompt':<25} | {'k':<2} | {'Tokens':<6} | {'Match?':<12} | {'Div Idx':<8} | {'Acc Rate':<8} | {'Speedup':<8}")
    print("-" * 90)
    for r in sweep_results:
        match_str = "YES" if r["exact_match"] else "NO"
        div_str = str(r["divergence_idx"]) if r["divergence_idx"] is not None else "-"
        acc_str = f"{r['acceptance_rate']}%"
        spd_str = f"{r['speedup']}x"
        print(f"{r['prompt']:<25} | {r['k']:<2} | {r['spec_len']:<6} | {match_str:<12} | {div_str:<8} | {acc_str:<8} | {spd_str:<8}")
    print("=" * 90)

    return sweep_results


if __name__ == "__main__":
    run_equivalence_sweeps()
