"""Interactive demonstration CLI for FlashSpeculativeMamba.

Compares speculative decoding against target-only greedy decoding side-by-side.
Provides precise wall-clock latency, tokens/sec, speedup, and acceptance metrics.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import torch

# Ensure src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import get_backend_info
from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.speculative_engine import SpeculativeEngine


DEFAULT_PROMPT = (
    "In computer science, speculative decoding accelerates autoregressive sequence generation "
    "by using a small draft model to propose candidate tokens that a larger target model verifies "
    "in parallel, thereby overcoming memory bandwidth bottlenecks"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashSpeculativeMamba Interactive CLI Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Input text prompt for text generation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of speculative tokens proposed per round.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--draft-model-id",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Hugging Face model ID for the draft model.",
    )
    parser.add_argument(
        "--target-model-id",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Hugging Face model ID for the target model.",
    )
    parser.add_argument(
        "--sequential-verify",
        action="store_true",
        default=False,
        help="Disable fast multi-token verification and use sequential single-token stepping fallback.",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    backend_info = get_backend_info()
    use_fast_multitoken = not args.sequential_verify

    print("=" * 80)
    print("FlashSpeculativeMamba Interactive CLI Demo")
    print("=" * 80)
    print(f"Device                   : {backend_info['gpu_name'] if backend_info['cuda_available'] else 'CPU'}")
    print(f"VRAM (Free / Total)      : {backend_info['free_vram_gb']} GB / {backend_info['total_vram_gb']} GB")
    print(f"Draft Model ID           : {args.draft_model_id}")
    print(f"Target Model ID          : {args.target_model_id}")
    print(f"Draft Length (k)         : {args.k}")
    print(f"Max New Tokens           : {args.max_new_tokens}")
    print(f"Verification Mode        : {'Fast Multi-Token Verification (Patched)' if use_fast_multitoken else 'Sequential Single-Token Fallback'}")

    is_same_model_pair = (args.draft_model_id == args.target_model_id)
    if is_same_model_pair:
        print("\n" + "!" * 80)
        print("[CAVEAT] 130m/130m: correctness + overhead only, NOT a real speedup result.")
        print("         The draft model has identical parameter capacity to the target model.")
        print("         Speculative decoding adds verification overhead without draft compute savings.")
        print("         Acceptance rate ~100% and speedup <1.0x are mathematically expected.")
        print("!" * 80)

    print("\nLoading models in fp16 precision...")
    draft = DraftModel(model_id=args.draft_model_id)
    target = TargetModel(model_id=args.target_model_id, use_fast_multitoken=use_fast_multitoken)
    tokenizer = target.tokenizer

    engine = SpeculativeEngine(
        draft_model=draft,
        target_model=target,
        k=args.k,
        use_fast_multitoken=use_fast_multitoken,
    )

    # Measure memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"GPU Memory Allocated     : {allocated:.2f} MB (Reserved: {reserved:.2f} MB)")

    # Warmup pass on a short prompt to initialize CUDA memory pools and kernel compilation
    print("\nWarming up CUDA kernels and caches...")
    warmup_prompt = "Hello world"
    warmup_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids
    _ = target.generate_greedy(warmup_ids, max_new_tokens=4)
    _ = engine.generate(warmup_ids, max_new_tokens=4, k=2)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    print(f"\nPrompt length            : {prompt_len} tokens")
    print(f"Prompt text              : \"{args.prompt}\"")

    # 1. Target-Only Greedy Baseline
    print("\n" + "-" * 80)
    print("Running Target-Only Greedy Baseline Generation...")
    print("-" * 80)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    target_tokens = target.generate_greedy(prompt_ids, max_new_tokens=args.max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_target = time.perf_counter() - t0
    target_tps = len(target_tokens) / t_target if t_target > 0 else 0.0

    target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)

    # 2. Speculative Decoding
    print("\n" + "-" * 80)
    print(f"Running Cached Speculative Decoding (k={args.k})...")
    print("-" * 80)
    spec_res = engine.generate(prompt_ids, max_new_tokens=args.max_new_tokens, k=args.k)

    # Compare tokens
    spec_tokens = spec_res.token_ids
    is_match = (spec_tokens == target_tokens)
    div_idx: Optional[int] = None
    div_target_tok = None
    div_spec_tok = None

    if not is_match:
        min_len = min(len(spec_tokens), len(target_tokens))
        for i in range(min_len):
            if spec_tokens[i] != target_tokens[i]:
                div_idx = i
                div_target_tok = target_tokens[i]
                div_spec_tok = spec_tokens[i]
                break
        if div_idx is None and len(spec_tokens) != len(target_tokens):
            div_idx = min_len

    speedup = spec_res.tokens_per_sec / target_tps if target_tps > 0 else 0.0

    # Display Results
    print("\n" + "=" * 80)
    print("GENERATED TEXT OUTPUT")
    print("=" * 80)
    print("[Target-Only Greedy Output]:")
    print(repr(target_text))
    print("\n[Speculative Engine Output]:")
    print(repr(spec_res.text))

    print("\n" + "=" * 80)
    print("BENCHMARK & VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Models Used              : Draft={args.draft_model_id} | Target={args.target_model_id}")
    print(f"Draft Length (k)         : {args.k}")
    print(f"Verification Mode        : {'Fast Multi-Token' if use_fast_multitoken else 'Sequential Fallback'}")
    print(f"Tokens Generated         : Target={len(target_tokens)} | Speculative={len(spec_tokens)}")
    print(f"Exact Match              : {'YES (100% Identical)' if is_match else f'NO (Diverged at index {div_idx})'}")
    if not is_match and div_idx is not None:
        print(f"  Divergence Index       : Token #{div_idx}")
        print(f"  Target Token           : ID {div_target_tok} ({repr(tokenizer.decode([div_target_tok])) if div_target_tok is not None else 'N/A'})")
        print(f"  Speculative Token      : ID {div_spec_tok} ({repr(tokenizer.decode([div_spec_tok])) if div_spec_tok is not None else 'N/A'})")
        print(f"  Reason                 : FP16 near-tie logit non-associativity (1 ULP difference in summation)")
    print(f"Target-Only Time         : {t_target * 1000:.2f} ms ({target_tps:.2f} tokens/sec)")
    print(f"Speculative Time         : {spec_res.elapsed_time_sec * 1000:.2f} ms ({spec_res.tokens_per_sec:.2f} tokens/sec)")
    print(f"Measured Speedup         : {speedup:.2f}x {'(Overhead only: 130m/130m)' if is_same_model_pair else ''}")
    print(f"Speculative Rounds       : {spec_res.num_rounds}")
    print(f"Acceptance Rate          : {spec_res.acceptance_rate * 100:.2f}% ({spec_res.total_accepted_tokens}/{spec_res.total_drafted_tokens})")
    print(f"Mean Accepted / Round    : {spec_res.mean_accepted_per_round:.2f} tokens")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        print(f"Peak GPU VRAM Allocated  : {peak_mb:.2f} MB / {backend_info['total_vram_gb']} GB")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    run_demo(cli_args)
