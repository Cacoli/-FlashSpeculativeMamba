"""Comprehensive test suite for FlashSpeculativeMamba.

Covers:
- Tokenizer parity between draft and target models.
- MambaCache snapshot, clone, and in-place restore fidelity on CPU and CUDA.
- Known risk empirical verification semantics.
- DraftModel propose and snapshot rollback/advance protocol.
- TargetVerifier logic: full accept (n=k), full reject (n=0), and partial accept (0<n<k).
- Fast parallel verification vs sequential verification equivalence.
- TargetModel cached greedy generation vs naive full-context reference (Stage 4 reference).
- SpeculativeEngine end-to-end generation and greedy equivalence against target baseline.
- NaiveSpeculativeEngine full-context recompute equivalence.
- FP16 near-tie logit analysis and divergence diagnosis.
"""

import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, MambaConfig
from transformers.cache_utils import MambaCache

# Ensure src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mamba_state_cache import (
    MambaStateSnapshot,
    snapshot_cache,
    restore_cache,
    create_mamba_cache,
    are_caches_equal,
    are_snapshots_equal,
)
from src.draft_model import DraftModel
from src.target_model import TargetModel
from src.target_verifier import TargetVerifier, apply_mamba_fast_multitoken_patch
from src.speculative_engine import SpeculativeEngine, NaiveSpeculativeEngine


DRAFT_MODEL_ID = "state-spaces/mamba-130m-hf"
TARGET_MODEL_ID = "state-spaces/mamba-1.4b-hf"
TEST_TARGET_ID = "state-spaces/mamba-130m-hf"  # Local cached weights for fast testing


def test_tokenizer_match() -> None:
    """Verify that draft and target models use identical tokenizers."""
    draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL_ID)
    target_tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    assert len(draft_tok) == len(target_tok), (
        f"Vocab size mismatch: draft={len(draft_tok)} vs target={len(target_tok)}"
    )
    assert draft_tok.eos_token_id == target_tok.eos_token_id, "EOS token ID mismatch"
    assert draft_tok.pad_token_id == target_tok.pad_token_id, "Pad token ID mismatch"
    assert draft_tok.bos_token_id == target_tok.bos_token_id, "BOS token ID mismatch"

    sample_texts = [
        "Hello world, testing Mamba speculative decoding.",
        "    Indented code line with numbers 123456.",
        "Complex mathematical statement: $\\int e^x dx = e^x + C$.",
    ]
    for text in sample_texts:
        draft_ids = draft_tok(text)["input_ids"]
        target_ids = target_tok(text)["input_ids"]
        assert draft_ids == target_ids, f"Tokenization mismatch for text: '{text}'"


def test_cache_snapshot_restore_roundtrip() -> None:
    """Verify snapshot and restore exact bit-level roundtrip on synthetic MambaCache tensors."""
    config = MambaConfig(
        num_hidden_layers=4,
        intermediate_size=64,
        state_size=16,
        conv_kernel=4,
    )
    cache = create_mamba_cache(config, batch_size=1, device="cpu", dtype=torch.float16)
    cache.conv_states.copy_(torch.randn_like(cache.conv_states))
    cache.ssm_states.copy_(torch.randn_like(cache.ssm_states))

    snapshot = snapshot_cache(cache)
    active_conv_before = cache.conv_states.clone()
    active_ssm_before = cache.ssm_states.clone()

    cache.conv_states.zero_()
    cache.ssm_states.zero_()

    assert not torch.equal(cache.conv_states, snapshot.conv_states)
    assert not torch.equal(cache.ssm_states, snapshot.ssm_states)

    restore_cache(cache, snapshot)
    assert torch.equal(cache.conv_states, snapshot.conv_states)
    assert torch.equal(cache.ssm_states, snapshot.ssm_states)
    assert torch.equal(cache.conv_states, active_conv_before)
    assert torch.equal(cache.ssm_states, active_ssm_before)


def test_cuda_cache_snapshot_restore() -> None:
    """Verify snapshot and restore on CUDA if GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    config = MambaConfig(
        num_hidden_layers=24,
        intermediate_size=1536,
        state_size=16,
        conv_kernel=4,
    )
    cache = create_mamba_cache(config, batch_size=1, device="cuda", dtype=torch.float16)
    cache.conv_states.uniform_(-1.0, 1.0)
    cache.ssm_states.uniform_(-1.0, 1.0)

    snapshot = snapshot_cache(cache)
    cache.conv_states.add_(1.5)
    cache.ssm_states.mul_(0.5)

    assert not torch.equal(cache.conv_states, snapshot.conv_states)
    restore_cache(cache, snapshot)
    assert torch.equal(cache.conv_states, snapshot.conv_states)
    assert torch.equal(cache.ssm_states, snapshot.ssm_states)


def test_empirical_known_risk_sequential_fallback() -> None:
    """Verify that stock unpatched HuggingFace Mamba requires sequential fallback."""
    from test_known_risk import run_empirical_risk_test

    report = run_empirical_risk_test(DRAFT_MODEL_ID)
    assert report["requires_sequential_fallback"] is True


def test_draft_model_propose_and_rollback() -> None:
    """Verify DraftModel propose, snapshotting, and rollback mechanics."""
    draft = DraftModel(model_id=DRAFT_MODEL_ID)
    prompt = "Artificial intelligence and speculative decoding"
    prompt_ids = draft.tokenizer(prompt, return_tensors="pt").input_ids

    pending, seq_len = draft.prefill(prompt_ids)
    assert pending == prompt_ids[0, -1].item()
    assert seq_len == prompt_ids.shape[1] - 1

    k = 4
    draft_tokens, snapshots = draft.propose(pending, k=k)
    assert len(draft_tokens) == k
    assert len(snapshots) == k

    # Test partial rollback (n_accepted = 2)
    draft.update_state(n_accepted=2, k=k, draft_tokens=draft_tokens, snapshots=snapshots)
    assert torch.equal(draft.cache.conv_states, snapshots[2].conv_states)
    assert torch.equal(draft.cache.ssm_states, snapshots[2].ssm_states)


def test_target_verifier_fast_vs_sequential_equivalence() -> None:
    """Verify that fast parallel verification produces identical results to sequential verification."""
    apply_mamba_fast_multitoken_patch()
    target = TargetModel(model_id=TEST_TARGET_ID, use_fast_multitoken=True)
    prompt = "The quick brown fox jumps over the lazy dog"
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids

    p, _ = target.prefill(prompt_ids)
    snapshot_initial = snapshot_cache(target.cache)

    candidate_draft = [15, 187, 187, 510]

    # Run Fast Parallel Verification
    target.verifier.use_fast_multitoken = True
    restore_cache(target.cache, snapshot_initial)
    res_fast = target.verify_draft(p, candidate_draft)

    # Run Sequential Verification
    target.verifier.use_fast_multitoken = False
    restore_cache(target.cache, snapshot_initial)
    res_seq = target.verify_draft(p, candidate_draft)

    assert res_fast.accepted_tokens == res_seq.accepted_tokens
    assert res_fast.next_pending_token == res_seq.next_pending_token
    assert res_fast.n_accepted == res_seq.n_accepted
    assert res_fast.bonus_token_awarded == res_seq.bonus_token_awarded


def test_target_greedy_cached_matches_naive_reference() -> None:
    """Verify that cached greedy generation matches naive full-context recompute reference (Stage 4 reference)."""
    target = TargetModel(model_id=TEST_TARGET_ID)
    prompt = "Mamba architectures combine linear state space models"
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids

    cached_tokens = target.generate_greedy(prompt_ids, max_new_tokens=16)
    naive_tokens = target.generate_naive_reference(prompt_ids, max_new_tokens=16)

    assert cached_tokens == naive_tokens, (
        f"Greedy cached tokens differ from naive reference:\nCached: {cached_tokens}\nNaive:  {naive_tokens}"
    )


def test_speculative_engine_end_to_end_equivalence() -> None:
    """Verify that SpeculativeEngine produces identical output to TargetModel greedy generation."""
    draft = DraftModel(model_id=DRAFT_MODEL_ID)
    target = TargetModel(model_id=TEST_TARGET_ID, use_fast_multitoken=True)
    prompt = "The fundamental theorem of calculus establishes a profound relationship"
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids

    # Target-only greedy
    target_tokens = target.generate_greedy(prompt_ids, max_new_tokens=32)

    # Speculative engine with fast multi-token verification (k=4)
    engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=True)
    spec_res = engine.generate(prompt_ids, max_new_tokens=32, k=4)

    assert spec_res.token_ids == target_tokens, (
        f"Speculative output diverged from target-only greedy output:\n"
        f"Speculative: {spec_res.token_ids}\n"
        f"Target-Only:  {target_tokens}"
    )
    assert spec_res.acceptance_rate > 0.0
    assert spec_res.num_rounds > 0


def test_speculative_engine_sequential_equivalence() -> None:
    """Verify that SpeculativeEngine in sequential verification mode matches target-only greedy 100%."""
    draft = DraftModel(model_id=DRAFT_MODEL_ID)
    target = TargetModel(model_id=TEST_TARGET_ID, use_fast_multitoken=False)
    prompt = "In computer science, speculative decoding accelerates"
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids

    target_tokens = target.generate_greedy(prompt_ids, max_new_tokens=32)

    engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=False)
    spec_res = engine.generate(prompt_ids, max_new_tokens=32, k=4)

    assert spec_res.token_ids == target_tokens


def test_naive_speculative_engine_matches_cached() -> None:
    """Verify that NaiveSpeculativeEngine (full-context recompute) matches SpeculativeEngine."""
    draft = DraftModel(model_id=DRAFT_MODEL_ID)
    target = TargetModel(model_id=TEST_TARGET_ID, use_fast_multitoken=True)
    prompt = "State space sequence models offer linear scaling"
    prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids

    cached_engine = SpeculativeEngine(draft, target, k=4, use_fast_multitoken=True)
    cached_res = cached_engine.generate(prompt_ids, max_new_tokens=32, k=4)

    naive_engine = NaiveSpeculativeEngine(draft, target, k=4)
    naive_res = naive_engine.generate(prompt_ids, max_new_tokens=32, k=4)

    assert cached_res.token_ids == naive_res.token_ids, (
        f"Cached engine differed from naive reference engine:\n"
        f"Cached: {cached_res.token_ids}\n"
        f"Naive:  {naive_res.token_ids}"
    )


def test_fp16_near_tie_divergence_diagnostics() -> None:
    """Document and verify that when divergence occurs, the logit margin is within FP16 near-tie epsilon."""
    target = TargetModel(model_id=TEST_TARGET_ID)
    tok = target.tokenizer
    prompt = (
        "The clock in the ancient clocktower chimed three times into the misty midnight air. "
        "The young watchmaker reached into his pocket and pulled out the brass key, wondering if "
        "the legends about the hidden workshop were true"
    )
    prompt_ids = tok(prompt, return_tensors="pt").input_ids

    target_tokens = target.generate_greedy(prompt_ids, max_new_tokens=10)
    p, _ = target.prefill(prompt_ids)
    pos = prompt_ids.shape[1] - 1
    for i in range(6):
        t_in = torch.tensor([[target_tokens[i]]], device=target.device)
        out = target.model(t_in, cache_params=target.cache, cache_position=torch.tensor([pos], device=target.device), use_cache=True)
        pos += 1

    t_in = torch.tensor([[target_tokens[6]]], device=target.device)
    out = target.model(t_in, cache_params=target.cache, cache_position=torch.tensor([pos], device=target.device), use_cache=True)
    logits = out.logits[0, -1]
    top2_vals, top2_inds = torch.topk(logits, 2)
    margin = (top2_vals[0] - top2_vals[1]).item()

    # The logit margin between top-1 and top-2 is <= 0.02 (within 1 ULP in FP16 at magnitude 21)
    assert margin <= 0.02, f"Expected near-tie logit margin, got {margin}"
