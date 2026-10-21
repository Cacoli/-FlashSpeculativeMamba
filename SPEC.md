# FlashSpeculativeMamba

Speculative decoding with Mamba (SSM) models. A small draft model proposes k tokens, the large target model verifies them in one pass, accepted tokens are kept, and the first wrong token is replaced by the target's own token.

## Environment & Hardware Specification
- **Project Root**: `C:\Users\ASUS\FlashSpeculativeMamba`
- **Python Environment**: `.venv` with Python 3.12.x (do NOT recreate or delete `.venv`; do NOT use Python 3.11 or 3.14).
- **PyTorch**: `2.11.0+cu128` installed from the official CUDA 12.8 index (intentionally omitted from `requirements.txt`; do NOT reinstall or overwrite torch).
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM, CUDA 12.8.
- **Model Precision**: `fp16` strictly for all model loads and inference. FP32 for 1.4B will cause CUDA OOM in 6 GB VRAM.
- **Package Management**: Use `.venv\Scripts\python.exe -m pip install ...` for any new packages and record them in `requirements.txt`.

## Goal
1. Greedy output matches target-only greedy output (report the mismatch rate, never assume 0).
2. Faster than target-only decoding in tokens/sec (report honestly if not).
3. benchmark.py measures both.

## Models
- Draft: state-spaces/mamba-130m-hf (loaded in fp16 only)
- Target: state-spaces/mamba-1.4b-hf (loaded in fp16 only; fallback: mamba-790m-hf)
- Same tokenizer, checked at startup, fail loudly if not.
- Greedy only. Sampling is a stretch goal.
- Precision: fp16 strictly (fp32 1.4B exceeds 6 GB VRAM when combined with draft and activations).

## Layout
FlashSpeculativeMamba/
  src/
    __init__.py
    speculative_engine.py
    draft_model.py
    target_model.py
    mamba_state_cache.py
    target_verifier.py
    backend.py
  demo.py
  benchmark.py
  test_speculative.py
  check_cuda.py
  requirements.txt
  README.md
  results/

## State convention
Mamba state per layer = conv state + SSM state. Fixed size, cannot be truncated like a KV cache, so rollback = restore a saved copy.
Each model's state covers all tokens BEFORE the pending token. The pending token is the last emitted token and is fed at the start of the next round.
Prefill: feed prompt[:-1]; the pending token is the last prompt token.

## One round (pending token p, draft length k)
1. Draft: feed p, get d1. Snapshot draft state after each feed: s0 (after p), s1 (after d1), ..., s(k-1). Generate d1..dk greedily. The draft never feeds dk during drafting.
2. Target: snapshot state T0. Feed [p, d1..dk] in ONE forward pass. logits[i] is the prediction after input i, i = 0..k.
3. Accept d_j while argmax(logits[j-1]) == d_j. Let n = number accepted (0..k).
4. New pending p' = argmax(logits[n]).
   - n == k: bonus token. Target state is already correct, no replay.
   - n < k: correction token. Restore T0, replay [p, d1..dn] to get the correct state.
5. Draft state: if n < k, restore s_n. If n == k, restore s(k-1) and feed dk once.
6. Emit d1..dn and p'. Repeat until EOS or max_new_tokens.
No full-context recompute after the initial prefill.

## Stage 4 reference
A naive version (target recomputes the full context each round) is built first and kept as the correctness reference. The cached engine must match it.

## Known risk to test before building on it
The verify step assumes one multi-token forward pass from a non-empty cache gives the same logits and final state as feeding tokens one at a time. Test this on the installed transformers version (fp16). If it fails, fall back to sequential verification and report it.

## Backend
backend.py and check_cuda.py report CUDA, GPU name, VRAM, and whether mamba-ssm / causal-conv1d import. If not, warn that the slow HF path is in use. Every benchmark row records the backend. Never compare speeds across backends.

## demo.py
Args: --prompt, --k (default 4), --max-new-tokens. Prints speculative output, target-only output, match yes/no, tokens/sec for both, acceptance rate.

## benchmark.py
20+ mixed prompts, fixed seed, warmup, torch.cuda.synchronize around timers, median over repeats. Sweeps: k in {2,4,6,8}, output length in {64,128,256,512}. Compare target-only vs naive speculative vs cached speculative. Log tokens/sec, speedup, acceptance rate, mean accepted per round, exact match, first divergence index, backend, GPU, dtype to results/benchmark.csv. Plots: speedup vs k, acceptance vs k, speedup vs length.

## test_speculative.py
Tokenizer check; snapshot/restore round-trip (torch.equal); k=1 and k=8; forced-reject rollback test; spec output == target-only output (log fp16 divergence); EOS and max_new_tokens edge cases.

## README must include
Why rollback is harder for SSMs, the round diagram, real results, backend used, limitations (greedy only, same family, numeric mismatches, slow path), positioning as implementation + empirical study, not a novelty claim.

## Stretch goals
1. Per-step target state caching to skip the replay pass.
2. Sampling via speculative rejection sampling.
3. Transformer-pair baseline with the same harness.

---

## Status Update: Stage 5 Complete (2026-10-21)
- `demo.py`: Fully interactive CLI demonstration supporting fast multi-token and sequential verification modes, side-by-side comparison with target-only baseline, exact match verification, and VRAM memory profiling.
- `benchmark.py`: Production benchmarking suite evaluating 20 prompts across lengths {64, 128, 256, 512} and draft lengths $k \in \{2, 4, 6, 8\}$. Generated 625 evaluation rows incrementally flushed to `results/benchmark.csv`.
- High-resolution plots generated and saved to `results/`:
  - `speedup_vs_k.png`
  - `acceptance_vs_k.png`
  - `speedup_vs_length.png`
- `README.md`: Comprehensive, publication-grade documentation containing architectural diagrams, mathematical proof of FP16 1-ULP non-associativity, 130m/130m benchmark caveat analysis, and viva defense notes.
- Test Suite: All 11 unit and equivalence tests in `test_speculative.py` pass (100%).

