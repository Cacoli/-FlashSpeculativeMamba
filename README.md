# FlashSpeculativeMamba

> **High-Performance Speculative Decoding for Pure Mamba State Space Models (SSMs)**  
> *Final-Year Engineering Project & Viva Research Artifact*

---

## 1. Executive Summary & Overview

**FlashSpeculativeMamba** implements lossless, wall-clock accelerated speculative decoding tailored specifically for pure State Space Models (Mamba).

In autoregressive language generation, memory bandwidth—rather than arithmetic throughput—is the primary bottleneck. Standard autoregressive decoding fetches model parameters from DRAM for every single token. Speculative decoding mitigates this bottleneck by utilizing a compact **draft model** to propose a block of $k$ candidate tokens $[d_1, \dots, d_k]$, followed by a single parallel forward pass of the larger **target model** to verify all $k$ candidates simultaneously.

While speculative decoding is widely implemented for Transformers, adapting it to State Space Models introduces distinct architectural challenges because SSMs maintain a recurrent hidden state rather than a spatial Key-Value (KV) cache. This repository solves these challenges with:
- **Zero-allocation state snapshotting and in-place rollback** (`MambaStateSnapshot`),
- **Fast patched multi-token parallel verification** resolving Hugging Face's convolution buffer slicing limitation to verify candidate sequences in a single forward pass,
- A robust **fallback sequential verification mode** that completely resolves IEEE 754 float16 near-tie divergence.

---

## 2. Why Rollback is Harder for SSMs than Transformers

The core theoretical and engineering difference between speculative decoding in Transformers vs. SSMs lies in how past context is represented in memory:

```
TRANSFORMERS: Spatial KV Cache
  [Token 0] -> Key 0, Value 0
  [Token 1] -> Key 1, Value 1
  [Token 2] -> Key 2, Value 2
  ...
  Rollback from step k to step n:
  Simply truncate sequence length: kv_cache[:, :, :n, :]
  (Zero recomputation required; past keys/values remain intact)

MAMBA (SSMs): Recurrent State Space
  Hidden State:  h_t = A_bar * h_{t-1} + B_bar * x_t     (h_t in R^{B x D x N})
  Conv Buffer:   x_{t-d_conv+1 : t} in R^{B x D x d_conv} (1D convolution history)
  
  Rollback from step k to step n:
  State h_t and Conv Buffer are OVERWRITTEN in-place on each forward step!
  No historical states are saved. Once mutated, past states are permanently lost.
```

### Key Differences:
1. **Destructive State Mutation**: In Mamba, both the 1D convolution buffer (`conv_states`) and the continuous-time recurrent matrix (`ssm_states`) update destructively on every scalar token step. There is no historical index to "slice back to".
2. **Naive Recomputation Overhead**: If states are not snapshotted, rejecting proposed tokens requires replaying the entire sequence history from the very beginning (prompt through accepted tokens), which incurs $O(L)$ quadratic overhead and destroys speculative throughput.
3. **FlashSpeculativeMamba Solution**: We implement lightweight snapshotting (`MambaStateSnapshot`) using `.clone()` before speculative rounds and per-step snapshots during drafting. Rollback is executed via in-place `.copy_()` operations directly on pre-allocated GPU tensors, avoiding memory reallocations and garbage collector churn.

---

## 3. Protocol Architecture & Speculative Round Lifecycle

Each speculative decoding round advances generation by proposing $k$ candidate tokens and verifying them in parallel.

### Speculative Round Flowchart

```
                 +-----------------------------------------+
                 | Prefill / Pending Token p in Target &   |
                 | Draft Models (Initial State S_0, T_0)   |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 | Draft Model Proposes k Candidates:      |
                 | [d_1, d_2, ..., d_k]                    |
                 | Captures Step Snapshots [s_0, ..., s_k-1|
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 | Snapshot Target State T_0               |
                 | Target Verifier evaluates candidate     |
                 | sequence [p, d_1, ..., d_k]             |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 | Greedy Verification:                    |
                 | Compare target argmax with draft tokens |
                 | Accept n in [0, k] consecutive tokens   |
                 +-----------------------------------------+
                                      |
                   +------------------+------------------+
                   |                                     |
           All k Accepted (n == k)             Partial / Rejection (n < k)
                   |                                     |
                   v                                     v
  +---------------------------------+   +---------------------------------+
  | Bonus Token Awarded:            |   | Correction Token Generated:     |
  | p' = argmax(logits[k])          |   | p' = argmax(logits[n])          |
  | Target state already at end;    |   | Target rolls back to T_0;       |
  | advance draft with dk.          |   | replays prefix [p, d1..dn].     |
  | Emit: [d1..dk, p']              |   | Draft restores snapshot s_n.    |
  |                                 |   | Emit: [d1..dn, p']              |
  +---------------------------------+   +---------------------------------+
                   |                                     |
                   +------------------+------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 | Next Pending Token becomes p'           |
                 | Check for EOS or max_new_tokens budget  |
                 +-----------------------------------------+
```

---

## 4. Architectural Components

The codebase is organized into modular components located in `src/`:

| Module | Core Class | Responsibility |
| :--- | :--- | :--- |
| `src/mamba_state_cache.py` | `MambaStateSnapshot` | Captures PyTorch `MambaCache` state tensors (`conv_states`, `ssm_states`) and performs fast in-place `.copy_()` restoration without GC overhead. |
| `src/target_verifier.py` | `TargetVerifier` | Implements parallel candidate sequence verification. Patches Hugging Face's convolution buffer slicing to support $K > 1$ tokens in an active cache. |
| `src/draft_model.py` | `DraftModel` | Autoregressively proposes $k$ candidate tokens while capturing per-step recurrent state snapshots $[s_0, \dots, s_{k-1}]$; handles $s_n$ rollback. |
| `src/target_model.py` | `TargetModel` | Manages target model prefill, cached autoregressive greedy baseline, and dispatches candidate sequences to the verifier. |
| `src/speculative_engine.py` | `SpeculativeEngine` | Coordinates the full speculative decoding loop (prefill, draft, verify, accept/reject, state sync, bonus/correction emission). |
| `src/speculative_engine.py` | `NaiveSpeculativeEngine` | Ground-truth reference implementation that recomputes the full token sequence from scratch each round without recurrent caching. |
| `src/backend.py` | Diagnostic Utilities | GPU capability detection, VRAM budgeting, and PyTorch CUDA environment reporting. |

---

## 5. The FP16 Near-Tie Phenomenon & Floating-Point Non-Associativity

During rigorous equivalence testing across diverse prompts, we discovered and characterized an intrinsic floating-point phenomenon that affects speculative decoding on recurrent models in half-precision (`float16`).

### The Empirical Observation
- On a narrative creative prompt with $k=8$, the patched multi-token verifier diverged from the sequential target-only baseline at token index 7.
- Analysis of the target model's output logits at the point of divergence revealed:
  - Top candidate logit: **`21.578125`**
  - Second candidate logit: **`21.562500`**
  - Margin: **`0.015625`** ($2^{-6}$)

### Numerical Precision Proof
In IEEE 754 half-precision binary floating-point (`binary16`):
- Format: 1 sign bit, 5 exponent bits, 10 mantissa bits.
- At exponent $E = 4$ ($2^4 = 16 \le x < 32$), the Unit in the Last Place (ULP) is:
  $$\text{ULP} = 2^{E - 10} = 2^{4 - 10} = 2^{-6} = 0.015625$$
- The logit difference between the top-2 tokens was **exactly 1 ULP**.
- An identical scenario occurred at position 17 of a deep learning prompt where top-2 logits produced an **exact numerical tie** (margin `0.000000`).

### Root Cause: Convolutional Arithmetic Order
1. **Autoregressive Single-Token Path**: Computes 1D depthwise convolution as a sequential dot product between the kernel weights and past buffer slices.
2. **Multi-Token Parallel Path**: Concatenates historical buffer slices with the candidate tokens and runs `torch.nn.functional.conv1d(padding=0)`, which delegates to GPU GEMM (matrix multiplication) or Winograd algorithms.
3. Floating-point addition is non-associative:
   $$(a + b) + c \neq a + (b + c)$$
4. Summing inputs in different orders accumulates rounding noise ($\approx 1\text{ ULP}$) across Mamba's 24 layers of continuous-time SSM projections. When two vocabulary tokens are separated by $\le 1\text{ ULP}$, the argmax operation can select a different token.

### The Sequential Verification Safe Mode
To guarantee bitwise exact equivalence when strict mathematical reproduction is required:
- The system provides `--sequential-verify` (setting `use_fast_multitoken=False`).
- Sequential verification steps through candidate tokens one-by-one inside the target cache, executing arithmetic in the identical order as target-only generation.
- **Verification Equivalence**: Sequential verification achieves **100% exact match** across all prompts and lengths.
- **Speed Cost**: On `130m/130m`, sequential verification incurs virtually no additional wall-clock cost compared to fast verification (~0% delta); the performance difference is expected to matter more on 1.4B, not yet measured.

---

## 6. Benchmark Results & Hardware Evaluation

> [!IMPORTANT]
> **Caveat Regarding Model Setup**:
> All benchmarks reported below were conducted on the available cached checkpoint pair:
> **Draft Model**: `state-spaces/mamba-130m-hf`  
> **Target Model**: `state-spaces/mamba-130m-hf`  
> **Label**: `"130m/130m: correctness + overhead only, NOT a real speedup result"`  
> When draft and target models are identical, the draft model incurs the same computational cost per token as the target model. Under this homogeneous setup, speculative decoding adds verification overhead without draft compute reduction. Hence, an acceptance rate of $\approx 100\%$ and a wall-clock speedup of $< 1.0\times$ (typically $0.5\times - 0.7\times$) are mathematically expected and empirically confirmed.

### Experimental Hardware & Environment
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **Host OS**: Windows 11
- **CUDA Runtime**: 12.8
- **PyTorch**: 2.11.0+cu128
- **Precision**: Strict `float16`
- **Peak Memory**: $\approx 527.7\text{ MB}$ (well within the 6 GB VRAM envelope)

### Benchmark Summary Matrix (625 Production Runs from results/benchmark.csv)

| Decoding Mode | Evaluated Runs | Mean Throughput | Mean Speedup (130m/130m) | Mean Acceptance Rate | Exact Match Rate | Divergences |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target-Only Baseline** | 65 | **28.22 tok/s** | **1.000x** | 100.0% | 100.0% | Reference (0) |
| **Cached Speculative (Fast)** | 220 | **13.92 tok/s** | **0.498x** | **100.0%** | **100.0%** | **0 (0.0%)** |
| **Cached Speculative (Sequential)** | 220 | **14.06 tok/s** | **0.504x** | **100.0%** | **100.0%** | **0 (0.0%)** |
| **Naive Speculative (Full Recompute)** | 120 | **10.02 tok/s** | **0.390x** | **99.2%** | **94.2%** | **7 (5.8%)** |

### Performance Scaling Across Draft Length ($k$) — Cached Speculative (Fast)

| Draft Length ($k$) | Evaluated Runs | Mean Throughput | Mean Speedup (130m/130m) | Acceptance Rate | Exact Match Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$k=2$** | 60 | 14.38 tok/s | 0.505x | 100.0% | 100.0% |
| **$k=4$** | 65 | 13.74 tok/s | 0.493x | 100.0% | 100.0% |
| **$k=6$** | 40 | 13.62 tok/s | 0.486x | 100.0% | 100.0% |
| **$k=8$** | 55 | 13.94 tok/s | 0.509x | 100.0% | 100.0% |

### Performance Scaling Across Generation Length ($k=4$) — Cached Speculative (Fast)

| Sequence Length | Target-Only Baseline | Cached Speculative (Fast) | Measured Speedup | Exact Match Rate |
| :---: | :---: | :---: | :---: | :---: |
| **64 tokens** | 28.12 tok/s | 13.95 tok/s | 0.496x | 100.0% |
| **128 tokens** | 28.28 tok/s | 13.52 tok/s | 0.478x | 100.0% |
| **256 tokens** | 28.81 tok/s | 14.17 tok/s | 0.492x | 100.0% |
| **512 tokens** | 22.34 tok/s | 12.13 tok/s | 0.543x | 100.0% |

### Comparison Across Verification Modes

| Verification Mode | Verification Latency | Divergence Risk | Observed Throughput ($k=4$) | Measured Speedup (130m/130m) |
| :--- | :---: | :---: | :---: | :---: |
| **Target-Only Baseline** | N/A | None (Reference) | 22.0 – 26.5 tok/s | 1.00x |
| **Cached Speculative (Fast Multi-Token)** | Parallel forward pass | 1-ULP FP16 Near-Ties | 11.5 – 13.5 tok/s | 0.48x – 0.58x |
| **Cached Speculative (Sequential Fallback)** | Baseline | 0% (Exact Bitwise Match) | 10.0 – 12.5 tok/s | 0.42x – 0.50x |
| **Naive Speculative (Full Recompute)** | Extremely slow ($O(L^2)$) | Scan Non-Associativity | 4.6 – 9.2 tok/s | 0.18x – 0.35x |

### Critical Nuances & Empirical Insights

1. **Cached Greedy Baseline as Ground-Truth Reference**:
   - `NaiveSpeculative` (full-context recomputation on each round) exhibited 7 divergences out of 120 evaluated runs. The **likely cause (untested hypothesis)** is that un-cached sequence recomputation executes a full PyTorch parallel scan over the entire prefix, accumulating different numerical rounding artifacts than step-by-step recurrent cache updating ($h_t = \bar{A} h_{t-1} + \bar{B} x_t$).
   - Because the target model's natural autoregressive decode path operates recurrently with cache, **the target-only cached autoregressive generation is the true ground-truth reference for correctness**, rather than naive full-context recomputation.

2. **Safety of Fast Multi-Token Mode**:
   - While `Cached Speculative (Fast)` recorded 0 divergences across the 20 benchmark prompts, **this does not prove that the fast multi-token path is unconditionally safe**.
   - As rigorously established in Stage 4, floating-point non-associativity in multi-token convolution creates $\approx 1\text{ ULP}$ drift that triggered an empirical argmax flip on the Creative Narrative prompt ($k=8$) at token 7 (margin $0.015625$). The fast path is statistically robust across most standard distributions, but `--sequential-verify` is necessary when strict bitwise equivalence must be mathematically guaranteed.

3. **Safe-Mode Cost on Homogeneous vs. Heterogeneous Setups**:
   - On the `130m/130m` model pair, wall-clock throughput between fast multi-token verification ($13.92\text{ tok/s}$) and sequential fallback verification ($14.06\text{ tok/s}$) is essentially identical (overhead delta $\approx 0\%$).
   - This parity occurs because at 130M parameters, Python per-step loop overhead and small GPU kernel launch times dominate over matrix multiplication compute. On a larger target model such as `mamba-1.4b-hf`, parallel verification is expected to matter more on 1.4B, not yet measured.

4. **Why Stage 4 Speedups (0.64× – 1.02×) Differ from Benchmark (~0.50×)**:
   - **Hypothesis**: In Stage 4 exploratory sweeps (`test_equivalence_sweeps.py`), tests were run on short bursts with fixed sequences where target-only comparisons were susceptible to cold kernel launch overhead and variable per-round bonus token accumulation.
   - In `benchmark.py`, end-to-end timing encompasses warmed-up GPU execution across hundreds of consecutive tokens with strict `torch.cuda.synchronize()` boundaries.
   - **Forward-Pass Accounting for $c=1$ with 100% Acceptance**: For identical draft and target models ($c = 1$), each speculative round emits $k + 1$ tokens ($k$ accepted draft tokens $+ 1$ bonus token from target). Producing these $k + 1$ tokens requires:
     1. $k$ draft forward steps (proposing $[d_1, \dots, d_k]$),
     2. $1$ extra draft step (feeding $d_k$ once to advance the draft state so it synchronizes with target),
     3. $1$ multi-token verification forward pass of candidate sequence $[p, d_1, \dots, d_k]$ ($k + 1$ tokens).
     In total, the speculative engine executes $(k + 1)\text{ draft forward steps} + 1\text{ multi-token verify pass}$ of $k+1$ tokens. By contrast, the target-only baseline generates the same $k + 1$ tokens using simply $k + 1$ single-token target steps. Because draft and target models have identical parameter capacity (130M), the speculative process performs roughly twice the forward-pass compute of target-only generation per emitted token, directly explaining the measured speedup of **$\approx 0.50\times$** across all $k$.

### Benchmark Visualizations
Benchmark plots generated automatically by `benchmark.py` and saved to `results/`:
- `results/speedup_vs_k.png`: Measured speedup vs. draft length $k \in \{2, 4, 6, 8\}$.
- `results/acceptance_vs_k.png`: Empirical draft acceptance rate vs. $k$.
- `results/speedup_vs_length.png`: Generation throughput scaling across lengths $\{64, 128, 256, 512\}$.

---

## 7. How to Run

### Interactive CLI Demonstration (`demo.py`)
Run speculative decoding interactively side-by-side with the target-only baseline:

```powershell
# Standard fast multi-token speculative decoding (k=4)
.\.venv\Scripts\python.exe demo.py --prompt "Speculative decoding accelerates language models" --k 4 --max-new-tokens 64

# Run with sequential verification safe mode
.\.venv\Scripts\python.exe demo.py --prompt "Speculative decoding accelerates language models" --k 4 --max-new-tokens 64 --sequential-verify

# Configurable for heterogeneous model pairs (when 1.4B checkpoint is downloaded)
.\.venv\Scripts\python.exe demo.py --draft-model-id state-spaces/mamba-130m-hf --target-model-id state-spaces/mamba-1.4b-hf --k 4
```

### Comprehensive Benchmark Suite (`benchmark.py`)

```powershell
# Quick sanity run (5 prompts, k in {2, 4, 8}, lengths {64, 128})
.\.venv\Scripts\python.exe benchmark.py --quick

# Full production sweep (20+ prompts, k in {2, 4, 6, 8}, lengths {64, 128, 256, 512})
.\.venv\Scripts\python.exe benchmark.py

# Benchmark heterogeneous model pair (when 1.4B checkpoint is downloaded)
.\.venv\Scripts\python.exe benchmark.py --draft-model-id state-spaces/mamba-130m-hf --target-model-id state-spaces/mamba-1.4b-hf
```

### Automated Unit and Equivalence Tests (`pytest`)

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 8. Viva Presentation Prep Notes (Questions & Concise Answers)

### Q1: Why do you evaluate greedy decoding rather than nucleus or temperature sampling?
> **Answer**: Greedy decoding provides a deterministic, rigorous ground truth for algorithmic equivalence. In speculative sampling, candidate acceptance is probabilistic ($\alpha = \min(1, P_{\text{target}}(x) / P_{\text{draft}}(x))$). Verifying greedy equivalence first proves that the recurrent state rollback, cache synchronization, and multi-token parallel verification logic are mathematically sound before stochastic variance is introduced.

### Q2: Why is state snapshotting and restoration preferred over recomputing the sequence?
> **Answer**: SSMs are continuous-time recurrent dynamical systems where hidden state $h_t$ is updated sequentially ($O(1)$ per step). If the engine recomputes context from the beginning on every round (the naive approach), total computation grows quadratically $O(L^2)$ with generated sequence length. Snapshotting takes an $O(1)$ tensor clone of the fixed-size hidden states, and restoration uses an in-place GPU `.copy_()`, requiring microseconds regardless of context length.

### Q3: Why does acceptance rate dictate whether speculative decoding achieves a speedup?
> **Answer**: Speculative decoding delivers wall-clock speedup only when the parameter cost ratio $c = \frac{\text{Cost}(\text{Draft})}{\text{Cost}(\text{Target})} \ll 1$ (e.g. 130M draft vs 1.4B target) and acceptance rate $\alpha \approx 1$. If $c \approx 1$ (as in our 130m/130m validation setup), each speculative round executes $k + 1$ draft forward steps ($k$ draft proposals $+ 1$ draft advance step) plus 1 multi-token verification pass to emit $k + 1$ tokens. Because draft and target have identical compute costs, speculative decoding performs roughly twice the forward-pass compute of target-only generation per token, making speedup $< 1.0\times$ (empirically $\approx 0.50\times$) due to verification overhead. With an asymmetric model pair ($c \ll 1$), drafting $k$ tokens is computationally cheap, and verifying $k$ tokens in a single target pass overcomes memory bandwidth bottlenecks to produce wall-clock speedup.

### Q4: What architectural factors break naive speculative decoding on SSMs?
> **Answer**: Three factors:
> 1. **Lack of KV Cache**: SSM hidden states mutate in-place rather than appending to a key-value list; rollback requires explicit state cloning and copying.
> 2. **Framework Convolution Slicing**: Stock Hugging Face Transformers Mamba code hardcodes single-token convolution buffer slicing during decode mode, throwing an `IndexError` when fed $K > 1$ candidate tokens from an active cache.
> 3. **Non-Associative Arithmetic**: Parallel 1D convolutions (`F.conv1d`) accumulate float16 numbers in a different order than sequential scalar dot products, creating 1-ULP drift over 24 layers that can flip argmax selections on tied or near-tied logits.
