# SPDX-License-Identifier: Apache-2.0
"""Cache forks must read the parent's attention KV in place and prefill in bounded chunks."""
from dataclasses import replace
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from qev import Qev
from qev.cache import (SharedPrefixCache, SharedPrefixLayer, _attention_forward, fork_inference_cache,
                       shared_prefix_attention)
from qev.encoding import Encoder, Limits
from qev.model import QevModel, ModelSpec
from qev.schema import Candidate, typed_record
from conftest import tiny_backbone

DEVICE = os.environ.get("QEV_TEST_DEVICE", "cpu")


@pytest.fixture(params=["none", "last-full-attention"])
def fork_model(request, tokenizer):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(5)
    config = tiny_backbone(len(tokenizer), hybrid=True).config
    # Two full-attention layers: forks of forks, and a joint layer after a shared one.
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype="fp32",
                     attention="sdpa", rows_per_forward=2, candidate_interaction=request.param)
    config._attn_implementation = "sdpa"
    model = QevModel(Qwen3_5TextModel(config), spec, tokenizer.pad_token_id).to(DEVICE).eval()
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.fill_(0.9)
    return model


def long_record(questions=2):
    q = {"color": {"type": "choice", "instructions": "Choose its color.",
                   "criteria": {"blue": "blue", "red": "a much longer red option", "green": "green"}},
         "known": {"type": "noul", "instructions": "Is the color stated?"}}
    return typed_record({"state": "The signal is blue, then it turns red. " * 3,
                         "questions": dict(list(q.items())[:questions])}, source="synthetic", record_id="long")


def attention_tensors(cache):
    return [t for layer in cache.layers if isinstance(layer, DynamicLayer) for t in (layer.keys, layer.values)]


def all_tensors(cache):
    return attention_tensors(cache) + [t for layer in cache.layers if not isinstance(layer, DynamicLayer)
                                       for t in (*layer.conv_states.values(), *layer.recurrent_states.values())]


def assert_features(a, b, atol=2e-5):
    for (qa, ca), (qb, cb) in zip(a, b, strict=True):
        torch.testing.assert_close(qa, qb, atol=atol, rtol=atol)
        torch.testing.assert_close(ca, cb, atol=atol, rtol=atol)


@pytest.mark.parametrize("past", [0, 2])
def test_shared_prefix_attention_matches_dense_attention(past):
    torch.manual_seed(0)
    rows, heads, kv_heads, new, dim = 3, 4, 2, 4, 16
    segments = [(torch.randn(1, kv_heads, n, dim, device=DEVICE), torch.randn(1, kv_heads, n, dim, device=DEVICE))
                for n in (5, 3)]
    layer = SharedPrefixLayer(segments)
    own = [torch.randn(rows, kv_heads, past + new, dim, device=DEVICE) for _ in range(2)]
    query = torch.randn(rows, new, heads, dim, device=DEVICE).transpose(1, 2)  # HF's non-contiguous layout
    config = SimpleNamespace(_attn_implementation="eager")
    with shared_prefix_attention(config, SharedPrefixCache([layer])):
        assert config._attn_implementation == "qev_shared_prefix"
        if past:
            layer.update(own[0][:, :, :past], own[1][:, :, :past])
        keys, values = layer.update(own[0][:, :, past:], own[1][:, :, past:])
        output, _ = _attention_forward(SimpleNamespace(layer_idx=0), query, keys, values, None, scaling=0.3)
    assert config._attn_implementation == "eager" and layer.get_seq_length() == 8 + past + new
    # Dense reference: every row sees both segments and its own earlier tokens, then causal new tokens.
    k = torch.cat([*(s[0].expand(rows, -1, -1, -1) for s in segments), own[0]], 2).repeat_interleave(2, 1)
    v = torch.cat([*(s[1].expand(rows, -1, -1, -1) for s in segments), own[1]], 2).repeat_interleave(2, 1)
    scores = query.double() @ k.double().transpose(-1, -2) * 0.3
    visible = torch.arange(k.shape[2], device=DEVICE)[None] <= torch.arange(new, device=DEVICE)[:, None] + 8 + past
    expected = (scores.masked_fill(~visible, float("-inf")).softmax(-1) @ v.double()).transpose(1, 2)
    torch.testing.assert_close(output.double(), expected, atol=1e-5, rtol=1e-5)


def test_forks_reference_parent_kv_and_leave_it_bit_identical(fork_model, encoder, monkeypatch):
    model = fork_model
    # Per-question forks; batched questions read the state cache without forks (test_batched_questions.py).
    monkeypatch.setattr(QevModel, "batches_questions", property(lambda self: False))
    enc = encoder(long_record())
    forks, states = [], []
    fork, extend = model.fork_cache, model.extend_state_cache

    def track_fork(cache, copies=1):
        forks.append((cache, fork(cache, copies)))
        return forks[-1][1]

    def track_state(*args):
        states.append(extend(*args))
        return states[-1]

    monkeypatch.setattr(model, "fork_cache", track_fork)
    monkeypatch.setattr(model, "extend_state_cache", track_state)
    features = model.cached_features(enc)
    state = states[0]
    snapshot = [t.clone() for t in all_tensors(state)]
    assert len(forks) == sum(1 + -(-len(q.candidates) // model.spec.rows_per_forward) for q in enc.questions)
    for parent, child in forks:
        assert type(child) is SharedPrefixCache
        for i, (source, layer) in enumerate(zip(parent.layers, child.layers, strict=True)):
            if not isinstance(layer, DynamicLayer):
                continue
            # Question forks read the state; candidate forks read the state and the question's own KV.
            expected = [*getattr(source, "segments", ()), (source.keys, source.values)]
            assert [tuple(map(id, s)) for s in layer.segments] == [tuple(map(id, s)) for s in expected]
            assert layer.segments[0][0] is state.layers[i].keys
            if not layer.is_initialized:  # candidate forks stop before the joint layer
                assert i == model.joint_layer and child.layers[1].is_initialized
                continue
            # A fork stores only the tokens it computed, never the prefix.
            assert layer.keys.shape[-2] == layer.get_seq_length() - layer.prefix_length
            assert layer.keys.shape[-2] <= max(len(c) for q in enc.questions for c in (q.prefix, *q.candidates))
    assert_features(features, model.cached_features(enc), atol=0)
    for before, after in zip(snapshot, all_tensors(state), strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_shared_prefix_matches_copied_forks(fork_model, encoder, monkeypatch, backend):
    model = fork_model
    model.backbone.config._attn_implementation = backend
    record = long_record()
    q = record.questions[0]
    q = replace(q, candidates=(*q.candidates, Candidate("x", "yet another, even longer option")))
    records = [encoder(replace(record, questions=(q, record.questions[1]))), encoder(replace(record, questions=(q,)))]
    shared = {}
    for rows in (1, 2, 32):
        model.spec.rows_per_forward = rows
        shared[rows] = [model.cached_features(enc) for enc in records]
    monkeypatch.setattr(QevModel, "shares_prefix", property(lambda self: False))
    copies = []
    original = model.fork_cache

    def track(cache, n=1):
        copies.append(original(cache, n))
        return copies[-1]

    monkeypatch.setattr(model, "fork_cache", track)
    for rows, features in shared.items():
        model.spec.rows_per_forward = rows
        for enc, expected in zip(records, features, strict=True):
            assert_features(expected, model.cached_features(enc))
    # Unaudited layouts keep full per-row copies and the backbone's own attention.
    assert copies and all(type(c) is DynamicCache for c in copies)
    assert model.backbone.config._attn_implementation == backend


def test_fork_cannot_run_under_other_attention(fork_model, encoder):
    model = fork_model
    enc = encoder(long_record(1))
    state = model.extend_state_cache(None, enc.state, 0)
    child = model.fork_cache(state, 2)
    with pytest.raises(RuntimeError, match="shared_prefix_attention"), torch.no_grad():
        model.backbone(input_ids=torch.tensor([enc.questions[0].prefix] * 2, device=model.device),
                       past_key_values=child, use_cache=True)
    with pytest.raises(ValueError, match="share_prefix"):
        fork_inference_cache(child, 1)


@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_chunked_prefill_matches_single_forward(fork_model, encoder, monkeypatch, chunk):
    model = fork_model
    enc, single = encoder(long_record()), encoder(long_record(1))
    whole = model.extend_state_cache(None, enc.state, 0)
    expected, expected_single = model.cached_features(enc), model.cached_features(single)
    calls = []
    original = model._run_rows

    def trace(rows, **kwargs):
        calls.append((rows, kwargs.get("offset", 0)))
        return original(rows, **kwargs)

    monkeypatch.setattr(model, "_run_rows", trace)
    model.prefill_chunk = chunk
    chunked = model.extend_state_cache(None, enc.state, 0)
    assert [offset for _, offset in calls] == list(range(0, len(enc.state), chunk))
    assert [list(r) for rows, _ in calls for r in rows] == [list(enc.state[i:i + chunk])
                                                            for i in range(0, len(enc.state), chunk)]
    for a, b in zip(all_tensors(whole), all_tensors(chunked), strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert_features(expected, model.cached_features(enc))
    assert_features(expected_single, model.cached_features(single))
    calls.clear()
    session = Qev(model, encoder).session(long_record().state)
    session.predict({"known": {"type": "noul", "instructions": "Is the color stated?"}})
    assert all(len(r) <= chunk for rows, offset in calls if offset < len(enc.state) for r in rows)
    assert session.cached_tokens == len(enc.state)


def test_raising_max_state_keeps_question_and_candidate_allowance():
    released = Limits(4096, 512, 256, 4096, 128)
    assert released.resized(50_000) == Limits(50_000, 512, 256, 50_000, 128)
    assert released.resized(50_000, 52_000).max_path == 52_000
    assert released.resized(None, 6000) == Limits(4096, 512, 256, 6000, 128)
    assert released.resized(1024) == Limits(1024, 512, 256, 4096, 128)


@pytest.mark.skipif(DEVICE != "cuda", reason="allocator accounting needs CUDA")
def test_question_memory_does_not_scale_with_state_and_candidates(tokenizer, monkeypatch):
    torch.manual_seed(5)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype="fp32",
                     attention="sdpa", rows_per_forward=8)
    backbone = tiny_backbone(len(tokenizer), hybrid=True)
    backbone.config._attn_implementation = "sdpa"
    model = QevModel(backbone, spec, tokenizer.pad_token_id).to("cuda").eval()
    encoder = Encoder(tokenizer, Limits(8192, 128, 128, 8192, 32))
    criteria = {str(i): f"option number {i}" for i in range(8)}

    def measure(repeats):
        enc = encoder(typed_record({"state": "The signal is blue. " * repeats, "questions": {
            "q": {"type": "choice", "instructions": "Choose one.", "criteria": criteria}}},
            source="synthetic", record_id="memory"))
        state = model.extend_state_cache(None, enc.state, 0)
        expected = model.state_cache_features(enc, state)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        assert_features(expected, model.state_cache_features(enc, state), atol=0)
        kv = sum(t.numel() * t.element_size() for t in attention_tensors(state))
        return torch.cuda.max_memory_allocated() - before, kv

    (short, short_kv), (long, long_kv) = measure(10), measure(200)
    # Copying forks grew by (1 + candidate rows) KV copies; reading in place adds no state-sized buffer.
    assert long - short < (long_kv - short_kv) / 4, (short, long, short_kv, long_kv)
    monkeypatch.setattr(QevModel, "shares_prefix", property(lambda self: False))
    (short, _), (long, _) = measure(10), measure(200)
    assert long - short > 8 * (long_kv - short_kv)
