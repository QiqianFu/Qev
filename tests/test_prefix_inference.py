# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Shared prefix execution must be real, branch-isolated, and numerically correct."""
from dataclasses import replace
import gc
import os
import weakref

import pytest
import torch

from qev.model import QevModel, ModelSpec
from qev.schema import Candidate
from conftest import tiny_backbone


@pytest.fixture(params=["none", "last-full-attention", "last-readout-cross"])
def prefix_model(request, tokenizer):
    torch.manual_seed(123)
    backbone = tiny_backbone(len(tokenizer), hybrid=True)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0,
                     weights_dtype="fp32", attention="eager", rows_per_forward=32,
                     candidate_interaction=request.param)
    model = QevModel(backbone, spec, tokenizer.pad_token_id)
    model.to(os.environ.get("QEV_TEST_DEVICE", "cpu")).eval()
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.fill_(0.8)
    return model


@pytest.mark.parametrize("single", [False, True])
def test_prefix_tokens_computed_once(prefix_model, encoder, record, monkeypatch, single):
    model = prefix_model
    if single:
        record = replace(record, questions=record.questions[:1])
    enc = encoder(record)
    reference = model.predict(enc, cached=False)
    calls = []
    original = model._run_rows

    def trace(rows, **kwargs):
        calls.append((rows, kwargs.get("offset", 0)))
        return original(rows, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached inference must not fall back to full leaf rows")

    monkeypatch.setattr(model, "_run_rows", trace)
    monkeypatch.setattr(model, "reference_features", forbidden)
    monkeypatch.setattr(model, "reference_batch_features", forbidden)
    result = model.predict(enc)
    for a, b in zip(reference, result, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    expected_tokens = len(enc.state) + sum(len(q.prefix) + sum(map(len, q.candidates)) for q in enc.questions)
    assert sum(len(row) for rows, _ in calls for row in rows) == expected_tokens
    assert len(calls) == (2 if single else 1 + 2 * len(enc.questions))
    if single:
        assert calls[0] == ([enc.state + enc.questions[0].prefix], 0)
    else:
        assert calls[0] == ([enc.state], 0)
    assert calls[-1][1] == len(enc.state) + len(enc.questions[-1].prefix)


@pytest.mark.parametrize("gate", [0.0, 1.3])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_joint_cached_features_with_unequal_lengths_and_chunking(prefix_model, encoder, record, gate, backend):
    model = prefix_model
    if model.joint_layer is None:
        pytest.skip("joint-specific readouts")
    model.backbone.config._attn_implementation = backend
    with torch.no_grad():
        model.joint_gate.fill_(gate)
    q = record.questions[0]
    q = replace(q, candidates=(q.candidates[0], Candidate("red", "a considerably longer red option"), q.candidates[2]))
    enc = encoder(replace(record, questions=(q, *record.questions[1:])))
    expected = model.reference_features(enc)
    for rows in (1, 2, 32):
        model.spec.rows_per_forward = rows
        for (q1, c1), (q2, c2) in zip(expected, model.cached_features(enc), strict=True):
            torch.testing.assert_close(q1, q2, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(c1, c2, atol=2e-5, rtol=2e-5)


def test_cached_branch_and_question_isolation(prefix_model, encoder, record):
    model = prefix_model
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.zero_()
    q = record.questions[0]
    changed = replace(q, candidates=(q.candidates[0], Candidate("red", "a much longer unrelated candidate"), q.candidates[2]))
    a = model.cached_features(encoder(record))
    b = model.cached_features(encoder(replace(record, questions=(changed, record.questions[1]))))
    torch.testing.assert_close(a[0][1][0], b[0][1][0], atol=2e-5, rtol=2e-5)
    for left, right in zip(a[1], b[1], strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.fill_(1.5)
        a = model.cached_features(encoder(record))
        b = model.cached_features(encoder(replace(record, questions=(changed, record.questions[1]))))
        assert (a[0][1][0] - b[0][1][0]).abs().max() > 1e-4
        torch.testing.assert_close(a[0][0], b[0][0], atol=2e-5, rtol=2e-5)
        for left, right in zip(a[1], b[1], strict=True):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


def test_single_candidate_and_candidate_token_budget(prefix_model, encoder, record, monkeypatch):
    model = prefix_model
    q = replace(record.questions[0], candidates=record.questions[0].candidates[:1], target=(1.0,))
    enc = encoder(replace(record, questions=(q,)))
    expected = model.reference_features(enc)
    for (q1, c1), (q2, c2) in zip(expected, model.cached_features(enc), strict=True):
        torch.testing.assert_close(q1, q2, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(c1, c2, atol=2e-5, rtol=2e-5)
    enc = encoder(record)
    model.spec.max_padded_tokens = max(len(c) for q in enc.questions for c in q.candidates)
    reference = model.predict(enc)
    original = model._run_rows
    seen = []

    def trace(rows, **kwargs):
        if len(rows) > 1:
            assert len(rows) * max(map(len, rows)) <= model.spec.max_padded_tokens
        seen.append(rows)
        return original(rows, **kwargs)

    monkeypatch.setattr(model, "_run_rows", trace)
    model.spec.rows_per_forward = 1
    for a, b in zip(reference, model.predict(enc), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert seen
    model.spec.max_padded_tokens = 1
    with pytest.raises(ValueError, match="candidate exceeds"):
        model.predict(enc)


def test_prediction_releases_cache_forks_without_cyclic_gc(prefix_model, encoder, record, monkeypatch):
    refs = []
    original = prefix_model.fork_cache

    def track(*args, **kwargs):
        cache = original(*args, **kwargs)
        refs.append(weakref.ref(cache))
        return cache

    monkeypatch.setattr(prefix_model, "fork_cache", track)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            prefix_model.predict(encoder(record))
        assert refs and all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
        gc.collect()
