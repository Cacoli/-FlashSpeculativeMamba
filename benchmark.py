"""Comprehensive benchmarking harness for FlashSpeculativeMamba.

Measures wall-clock throughput (tokens/sec), speedup, acceptance rates, and
greedy equivalence across multiple prompt lengths, draft lengths (k), and
decoding modes:
  1. Target-Only (recurrent cached greedy baseline)
  2. Naive Speculative (full context recompute reference)
  3. Cached Speculative (Fast Multi-Token Verification via patched conv/SSM)
  4. Cached Speculative (Sequential Single-Token Verification fallback)

Outputs:
  - CSV log: results/benchmark.csv (written incrementally)
  - Matplotlib plots:
      - results/speedup_vs_k.png
      - results/acceptance_vs_k.png
      - results/speedup_vs_length.png
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import get_backend_info
from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.speculative_engine import SpeculativeEngine, NaiveSpeculativeEngine, SpeculativeGenerationResult


# Standard benchmark prompts spanning multiple domains (22 total)
BENCHMARK_PROMPTS = [
    {
        "id": "p01_cs_speculative",
        "category": "Computer Science",
        "text": (
            "In computer science, speculative decoding accelerates autoregressive sequence generation "
            "by using a small draft model to propose candidate tokens that a larger target model verifies "
            "in parallel, thereby overcoming memory bandwidth bottlenecks"
        ),
    },
    {
        "id": "p02_dl_transformers",
        "category": "Deep Learning",
        "text": (
            "Deep learning architectures have revolutionized natural language processing, computer vision, "
            "and speech recognition by leveraging multi-layer neural networks"
        ),
    },
    {
        "id": "p03_thermodynamics",
        "category": "Physics",
        "text": (
            "The thermodynamic principles governing heat transfer, entropy generation, and energy conservation "
            "state that in any isolated system spontaneous processes increase total entropy"
        ),
    },
    {
        "id": "p04_dna_transcription",
        "category": "Biology",
        "text": (
            "The cellular mechanisms underlying DNA transcription, RNA splicing, and protein synthesis in "
            "eukaryotic cells involve complex enzymatic assemblies"
        ),
    },
    {
        "id": "p05_maritime_law",
        "category": "Law",
        "text": (
            "Under international maritime law, vessels navigating international straits and territorial waters "
            "are granted the right of transit passage provided they proceed without delay"
        ),
    },
    {
        "id": "p06_monetary_policy",
        "category": "Economics",
        "text": (
            "The economic impact of global monetary policy, inflation targeting, and central bank interest rate "
            "adjustments creates systemic changes in capital allocation"
        ),
    },
    {
        "id": "p07_creative_clockmaker",
        "category": "Narrative Fiction",
        "text": (
            "Once upon a time in a forgotten forest enveloped by perpetual mist, an ancient clockmaker built "
            "a mechanical hummingbird capable of weaving memories from silver thread"
        ),
    },
    {
        "id": "p08_quantum_mechanics",
        "category": "Quantum Physics",
        "text": (
            "Quantum mechanical systems exhibit wave-particle duality, superposition of orthogonal eigenstates, "
            "and non-local entanglement between spatially separated particles"
        ),
    },
    {
        "id": "p09_renaissance_history",
        "category": "History",
        "text": (
            "The Renaissance period witnessed profound artistic and philosophical transformations across "
            "Western Europe, characterized by humanism and revived classical antiquity"
        ),
    },
    {
        "id": "p10_distributed_systems",
        "category": "Systems Engineering",
        "text": (
            "Modern distributed database systems maintain consistency and partition tolerance through "
            "consensus algorithms such as Raft and Paxos, ensuring high availability"
        ),
    },
    {
        "id": "p11_photosynthesis",
        "category": "Biochemistry",
        "text": (
            "Photosynthesis is the biochemical process through which green plants, algae, and cyanobacteria "
            "convert solar light energy into chemical energy stored in carbohydrates"
        ),
    },
    {
        "id": "p12_contract_law",
        "category": "Jurisprudence",
        "text": (
            "In contract law, the fundamental prerequisites for creating an enforceable agreement comprise "
            "mutual assent, valid consideration, contractual capacity, and legal legality"
        ),
    },
    {
        "id": "p13_semiconductor_supply",
        "category": "Geopolitics & Tech",
        "text": (
            "The geopolitical dynamics of renewable energy transition, critical mineral supply chains, "
            "and semiconductor manufacturing have restructured international commerce"
        ),
    },
    {
        "id": "p14_synaptic_plasticity",
        "category": "Neuroscience",
        "text": (
            "Neuroscientists studying synaptic plasticity and long-term potentiation have demonstrated that "
            "persistent strengthening of synapses leads to lasting changes in signal transmission"
        ),
    },
    {
        "id": "p15_calculus_theorem",
        "category": "Mathematics",
        "text": (
            "A fundamental theorem of calculus establishes the formal relationship between differentiation "
            "and integration, stating that if a real function is continuous on an interval"
        ),
    },
    {
        "id": "p16_industrial_revolution",
        "category": "Economic History",
        "text": (
            "During the Industrial Revolution, the advent of steam power, mechanized textile manufacturing, "
            "and metallurgical innovations radically altered agrarian societies"
        ),
    },
    {
        "id": "p17_epistemology",
        "category": "Philosophy",
        "text": (
            "The philosophical foundations of epistemology explore the nature, origin, and limits of human "
            "knowledge, addressing skepticism through foundationalism and coherentism"
        ),
    },
    {
        "id": "p18_hpc_memory",
        "category": "Computer Architecture",
        "text": (
            "In high-performance computing, optimizing memory hierarchy access patterns, cache coherence, "
            "and parallel execution pipelines is critical to overcome memory walls"
        ),
    },
    {
        "id": "p19_marine_coral_ecology",
        "category": "Ecology",
        "text": (
            "The ecology of marine coral reefs depends on delicate symbiotic relationships between "
            "dinoflagellate algae and calcifying anthozoan polyps in shallow tropical waters"
        ),
    },
    {
        "id": "p20_cryptographic_primitives",
        "category": "Cryptography",
        "text": (
            "Cryptographic primitives such as elliptic curve digital signatures, zero-knowledge proofs, "
            "and collision-resistant hash functions provide the foundation for decentralized ledgers"
        ),
    },
    {
        "id": "p21_aerospace_propulsion",
        "category": "Aerospace",
        "text": (
            "The development of aerospace propulsion systems, from turbofan engines to hypersonic scramjets "
            "and ion thrusters, requires balancing thermal efficiency and structural integrity"
        ),
    },
    {
        "id": "p22_evolutionary_selection",
        "category": "Evolutionary Biology",
        "text": (
            "In evolutionary biology, natural selection acts upon heritable phenotypic variation within "
            "populations, favoring adaptations that enhance differential reproductive success"
        ),
    },
]

CSV_FIELDNAMES = [
    "model_pair",
    "k",
    "length",
    "prompt_id",
    "mode",
    "tokens_per_sec",
    "speedup",
    "acceptance_rate",
    "exact_match",
    "divergence_index",
    "actual_length",
    "elapsed_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashSpeculativeMamba Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--draft-model-id",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Draft Hugging Face model identifier.",
    )
    parser.add_argument(
        "--target-model-id",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Target Hugging Face model identifier.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sanity benchmark: 5 prompts, k in {2, 4, 8}, lengths {64, 128}.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit number of prompts to run (default: 5 if --quick else 20).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeat generation runs per configuration (records median).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save benchmark CSV and figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark execution.",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Incrementally append a single result row to CSV and flush to disk."""
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def run_benchmark(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"

    # Reset or initialize CSV
    if csv_path.exists():
        csv_path.unlink()
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

    backend_info = get_backend_info()
    model_pair = f"{Path(args.draft_model_id).name}/{Path(args.target_model_id).name}"
    is_same_model = (args.draft_model_id == args.target_model_id)

    print("=" * 80)
    print("FlashSpeculativeMamba Comprehensive Benchmarking Suite")
    print("=" * 80)
    print(f"Device                   : {backend_info['gpu_name'] if backend_info['cuda_available'] else 'CPU'}")
    print(f"CUDA Available           : {backend_info['cuda_available']} (v{backend_info['cuda_version']})")
    print(f"Model Pair               : {model_pair}")
    print(f"Draft Model ID           : {args.draft_model_id}")
    print(f"Target Model ID          : {args.target_model_id}")
    print(f"Mode                     : {'QUICK SANITY RUN' if args.quick else 'FULL PRODUCTION BENCHMARK'}")
    print(f"Repeats per test         : {args.repeats}")
    print(f"Output CSV               : {csv_path}")

    if is_same_model:
        caveat_label = "mamba-130m-hf / mamba-130m-hf | 130m/130m: correctness + overhead only, NOT a real speedup result"
        print("\n" + "!" * 80)
        print(f"[CAVEAT] {caveat_label}")
        print("         The draft model is identical to the target model.")
        print("         Speculative decoding adds verification overhead without compute reduction.")
        print("         Acceptance rate ~100% and speedup <1.0x are mathematically expected.")
        print("!" * 80 + "\n")
    else:
        caveat_label = f"{model_pair} real speculative decoding"

    # Define benchmark matrix
    if args.quick:
        num_prompts = args.max_prompts or 5
        k_values = [2, 4, 8]
        lengths = [64, 128]
        include_512 = False
        naive_max_len = 128
    else:
        num_prompts = args.max_prompts or 20
        k_values = [2, 4, 6, 8]
        lengths = [64, 128, 256]
        include_512 = True
        naive_max_len = 128  # Naive capped at 128 to prevent extreme quadratic slowdown

    prompts_to_run = BENCHMARK_PROMPTS[:num_prompts]
    print(f"Selected {len(prompts_to_run)} prompts for benchmarking.")
    print(f"Draft k values           : {k_values}")
    print(f"Base generation lengths  : {lengths}")
    print(f"512 length enabled       : {include_512} (Target-Only & Cached Speculative only)")
    print(f"NaiveSpeculative Cap     : max length {naive_max_len} (capped to prevent quadratic slowdown)")

    # Load models
    print("\nLoading models in fp16 precision...")
    draft = DraftModel(model_id=args.draft_model_id)
    target = TargetModel(model_id=args.target_model_id, use_fast_multitoken=True)
    tokenizer = target.tokenizer

    cached_fast_engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=True)
    cached_seq_engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=False)
    naive_engine = NaiveSpeculativeEngine(draft, target, k=4)

    # Warmup pass
    print("Warming up CUDA kernels and caching allocator...")
    warmup_ids = tokenizer("Warmup prompt for GPU allocation", return_tensors="pt").input_ids
    _ = target.generate_greedy(warmup_ids, max_new_tokens=4, eos_token_id=-1)
    _ = cached_fast_engine.generate(warmup_ids, max_new_tokens=4, k=2, eos_token_id=-1)

    all_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("STARTING BENCHMARK EXECUTION")
    print("=" * 80)

    for p_idx, p_info in enumerate(prompts_to_run, start=1):
        p_id = p_info["id"]
        p_cat = p_info["category"]
        p_text = p_info["text"]
        prompt_ids = tokenizer(p_text, return_tensors="pt").input_ids

        # Prompts 1..5 run 512 if enabled; others run base lengths
        target_lengths = list(lengths)
        if include_512 and p_idx <= 5:
            target_lengths.append(512)

        for max_len in target_lengths:
            print(f"\n[Prompt {p_idx}/{len(prompts_to_run)}] '{p_id}' ({p_cat}) | Length: {max_len}")

            # -------------------------------------------------------------
            # 1. Target-Only Baseline (Ground Truth)
            # -------------------------------------------------------------
            target_tps_runs: List[float] = []
            target_time_runs: List[float] = []
            target_tokens: List[int] = []

            for _ in range(args.repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                toks = target.generate_greedy(prompt_ids, max_new_tokens=max_len, eos_token_id=-1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_el = time.perf_counter() - t0
                target_tokens = toks
                tps = len(toks) / t_el if t_el > 0 else 0.0
                target_tps_runs.append(tps)
                target_time_runs.append(t_el)

            target_tps_med = sorted(target_tps_runs)[len(target_tps_runs) // 2]
            target_time_med = sorted(target_time_runs)[len(target_time_runs) // 2]

            rec_target = {
                "model_pair": caveat_label,
                "k": 1,
                "length": max_len,
                "prompt_id": p_id,
                "mode": "Target-Only",
                "tokens_per_sec": round(target_tps_med, 2),
                "speedup": 1.0,
                "acceptance_rate": 1.0,
                "exact_match": True,
                "divergence_index": "",
                "actual_length": len(target_tokens),
                "elapsed_time_sec": round(target_time_med, 4),
            }
            append_csv_row(csv_path, rec_target)
            all_records.append(rec_target)
            print(f"  Target-Only                  : {target_tps_med:6.2f} tok/s (1.00x) | gen={len(target_tokens)}")

            # Select k values for this length
            # For 512: run canonical k in {4, 8} to avoid excessive generation time
            active_k_values = [4, 8] if max_len == 512 else k_values

            # -------------------------------------------------------------
            # 2. Speculative Modes across active_k_values
            # -------------------------------------------------------------
            for k in active_k_values:
                # A. Cached Speculative (Fast Multi-Token Verification)
                fast_tps_runs: List[float] = []
                fast_res: Optional[SpeculativeGenerationResult] = None

                for _ in range(args.repeats):
                    res = cached_fast_engine.generate(prompt_ids, max_new_tokens=max_len, k=k, eos_token_id=-1)
                    fast_tps_runs.append(res.tokens_per_sec)
                    fast_res = res

                assert fast_res is not None
                med_fast_tps = sorted(fast_tps_runs)[len(fast_tps_runs) // 2]
                fast_speedup = med_fast_tps / target_tps_med if target_tps_med > 0 else 0.0
                match_fast = (fast_res.token_ids == target_tokens)
                div_fast = ""
                if not match_fast:
                    min_l = min(len(fast_res.token_ids), len(target_tokens))
                    for idx in range(min_l):
                        if fast_res.token_ids[idx] != target_tokens[idx]:
                            div_fast = str(idx)
                            break
                    if not div_fast and len(fast_res.token_ids) != len(target_tokens):
                        div_fast = str(min_l)
                    print(f"    [DIVERGENCE] Fast Multi-Token k={k} diverged at token #{div_fast} on {p_id}")

                rec_fast = {
                    "model_pair": caveat_label,
                    "k": k,
                    "length": max_len,
                    "prompt_id": p_id,
                    "mode": "Cached Speculative (Fast)",
                    "tokens_per_sec": round(med_fast_tps, 2),
                    "speedup": round(fast_speedup, 3),
                    "acceptance_rate": round(fast_res.acceptance_rate, 4),
                    "exact_match": match_fast,
                    "divergence_index": div_fast,
                    "actual_length": len(fast_res.token_ids),
                    "elapsed_time_sec": round(fast_res.elapsed_time_sec, 4),
                }
                append_csv_row(csv_path, rec_fast)
                all_records.append(rec_fast)
                print(
                    f"  Cached Spec (Fast)   k={k}    : {med_fast_tps:6.2f} tok/s ({fast_speedup:4.2f}x) | "
                    f"acc={fast_res.acceptance_rate * 100:5.1f}% | match={'YES' if match_fast else 'DIV@' + div_fast}"
                )

                # B. Cached Speculative (Sequential Verification Fallback)
                seq_tps_runs: List[float] = []
                seq_res: Optional[SpeculativeGenerationResult] = None

                for _ in range(args.repeats):
                    res = cached_seq_engine.generate(prompt_ids, max_new_tokens=max_len, k=k, eos_token_id=-1)
                    seq_tps_runs.append(res.tokens_per_sec)
                    seq_res = res

                assert seq_res is not None
                med_seq_tps = sorted(seq_tps_runs)[len(seq_tps_runs) // 2]
                seq_speedup = med_seq_tps / target_tps_med if target_tps_med > 0 else 0.0
                match_seq = (seq_res.token_ids == target_tokens)
                div_seq = ""
                if not match_seq:
                    min_l = min(len(seq_res.token_ids), len(target_tokens))
                    for idx in range(min_l):
                        if seq_res.token_ids[idx] != target_tokens[idx]:
                            div_seq = str(idx)
                            break
                    if not div_seq and len(seq_res.token_ids) != len(target_tokens):
                        div_seq = str(min_l)
                    print(f"    [DIVERGENCE] Sequential Fallback k={k} diverged at token #{div_seq} on {p_id}")

                rec_seq = {
                    "model_pair": caveat_label,
                    "k": k,
                    "length": max_len,
                    "prompt_id": p_id,
                    "mode": "Cached Speculative (Sequential)",
                    "tokens_per_sec": round(med_seq_tps, 2),
                    "speedup": round(seq_speedup, 3),
                    "acceptance_rate": round(seq_res.acceptance_rate, 4),
                    "exact_match": match_seq,
                    "divergence_index": div_seq,
                    "actual_length": len(seq_res.token_ids),
                    "elapsed_time_sec": round(seq_res.elapsed_time_sec, 4),
                }
                append_csv_row(csv_path, rec_seq)
                all_records.append(rec_seq)
                print(
                    f"  Cached Spec (Seq)    k={k}    : {med_seq_tps:6.2f} tok/s ({seq_speedup:4.2f}x) | "
                    f"acc={seq_res.acceptance_rate * 100:5.1f}% | match={'YES' if match_seq else 'DIV@' + div_seq}"
                )

                # C. Naive Speculative (Recompute Context Reference)
                # Capped at naive_max_len (128) and only on first 5 prompts for k in {2, 4, 8} to avoid extreme quadratic slowdown
                run_naive = (max_len <= naive_max_len) and (p_idx <= 5 or k == 4) and (k != 6)
                if run_naive:
                    naive_tps_runs: List[float] = []
                    naive_res: Optional[SpeculativeGenerationResult] = None

                    for _ in range(args.repeats):
                        res = naive_engine.generate(prompt_ids, max_new_tokens=max_len, k=k, eos_token_id=-1)
                        naive_tps_runs.append(res.tokens_per_sec)
                        naive_res = res

                    assert naive_res is not None
                    med_naive_tps = sorted(naive_tps_runs)[len(naive_tps_runs) // 2]
                    naive_speedup = med_naive_tps / target_tps_med if target_tps_med > 0 else 0.0
                    match_naive = (naive_res.token_ids == target_tokens)
                    div_naive = ""
                    if not match_naive:
                        min_l = min(len(naive_res.token_ids), len(target_tokens))
                        for idx in range(min_l):
                            if naive_res.token_ids[idx] != target_tokens[idx]:
                                div_naive = str(idx)
                                break
                        if not div_naive and len(naive_res.token_ids) != len(target_tokens):
                            div_naive = str(min_l)
                        print(f"    [DIVERGENCE] Naive Speculative k={k} diverged at token #{div_naive} on {p_id}")

                    rec_naive = {
                        "model_pair": caveat_label,
                        "k": k,
                        "length": max_len,
                        "prompt_id": p_id,
                        "mode": "Naive Speculative",
                        "tokens_per_sec": round(med_naive_tps, 2),
                        "speedup": round(naive_speedup, 3),
                        "acceptance_rate": round(naive_res.acceptance_rate, 4),
                        "exact_match": match_naive,
                        "divergence_index": div_naive,
                        "actual_length": len(naive_res.token_ids),
                        "elapsed_time_sec": round(naive_res.elapsed_time_sec, 4),
                    }
                    append_csv_row(csv_path, rec_naive)
                    all_records.append(rec_naive)
                    print(
                        f"  Naive Spec (Recomp)  k={k}    : {med_naive_tps:6.2f} tok/s ({naive_speedup:4.2f}x) | "
                        f"acc={naive_res.acceptance_rate * 100:5.1f}% | match={'YES' if match_naive else 'DIV@' + div_naive}"
                    )

    # Generate Plots
    print("\nGenerating benchmark plots from complete records...")
    df = pd.DataFrame(all_records)
    generate_plots(df, out_dir, caveat_label)
    print("Benchmark complete! All artifacts saved.")


def generate_plots(df: pd.DataFrame, out_dir: Path, caveat_label: str) -> None:
    """Generate high-resolution benchmark plots using matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    spec_df = df[df["mode"] != "Target-Only"].copy()

    colors = {
        "Cached Speculative (Fast)": "#1f77b4",
        "Cached Speculative (Sequential)": "#e07a5f",
        "Naive Speculative": "#2ca02c",
    }
    markers = {
        "Cached Speculative (Fast)": "o",
        "Cached Speculative (Sequential)": "s",
        "Naive Speculative": "^",
    }

    # 1. Plot: Speedup vs k
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    df_k = spec_df.groupby(["mode", "k"])["speedup"].mean().reset_index()

    for mode in ["Cached Speculative (Fast)", "Cached Speculative (Sequential)", "Naive Speculative"]:
        sub = df_k[df_k["mode"] == mode].sort_values("k")
        if not sub.empty:
            ax.plot(
                sub["k"],
                sub["speedup"],
                marker=markers.get(mode, "o"),
                linewidth=2.2,
                markersize=7,
                color=colors.get(mode, "#333333"),
                label=mode,
            )

    ax.axhline(1.0, color="#d62828", linestyle="--", linewidth=1.5, label="Target-Only Baseline (1.00x)")
    ax.set_title(
        f"Speedup vs. Draft Length (k)\n[{caveat_label}]",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Draft Length (k)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Measured Speedup (tokens/sec ratio)", fontsize=10, fontweight="bold")
    ax.set_xticks(sorted(df_k["k"].unique()))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="best")
    plt.tight_layout()
    p1 = out_dir / "speedup_vs_k.png"
    plt.savefig(p1)
    plt.close()
    print(f"Saved plot: {p1}")

    # 2. Plot: Acceptance Rate vs k
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    df_acc = spec_df.groupby(["mode", "k"])["acceptance_rate"].mean().reset_index()

    for mode in ["Cached Speculative (Fast)", "Cached Speculative (Sequential)", "Naive Speculative"]:
        sub = df_acc[df_acc["mode"] == mode].sort_values("k")
        if not sub.empty:
            ax.plot(
                sub["k"],
                sub["acceptance_rate"] * 100.0,
                marker=markers.get(mode, "o"),
                linewidth=2.2,
                markersize=7,
                color=colors.get(mode, "#333333"),
                label=mode,
            )

    ax.set_title(
        f"Acceptance Rate vs. Draft Length (k)\n[{caveat_label}]",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Draft Length (k)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Acceptance Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(sorted(df_acc["k"].unique()))
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    plt.tight_layout()
    p2 = out_dir / "acceptance_vs_k.png"
    plt.savefig(p2)
    plt.close()
    print(f"Saved plot: {p2}")

    # 3. Plot: Speedup vs Generation Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    df_len = spec_df[spec_df["k"] == 4].groupby(["mode", "length"])["speedup"].mean().reset_index()

    for mode in ["Cached Speculative (Fast)", "Cached Speculative (Sequential)", "Naive Speculative"]:
        sub = df_len[df_len["mode"] == mode].sort_values("length")
        if not sub.empty:
            ax.plot(
                sub["length"],
                sub["speedup"],
                marker=markers.get(mode, "o"),
                linewidth=2.2,
                markersize=7,
                color=colors.get(mode, "#333333"),
                label=f"{mode} (k=4)",
            )

    ax.axhline(1.0, color="#d62828", linestyle="--", linewidth=1.5, label="Target-Only Baseline (1.00x)")
    ax.set_title(
        f"Speedup vs. Generation Length (k=4)\n[{caveat_label}]",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Generated Sequence Length (tokens)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Measured Speedup (tokens/sec ratio)", fontsize=10, fontweight="bold")
    lengths_present = sorted(df_len["length"].unique())
    ax.set_xticks(lengths_present)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="best")
    plt.tight_layout()
    p3 = out_dir / "speedup_vs_length.png"
    plt.savefig(p3)
    plt.close()
    print(f"Saved plot: {p3}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
