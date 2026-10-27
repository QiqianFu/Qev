# SPDX-License-Identifier: Apache-2.0
"""Session state extension must reuse the state cache and match full recomputation."""
import os

import pytest
import torch

from qev import Qev
from qev.encoding import ContextOverflow
from qev.model import QevModel, ModelSpec
from qev.schema import typed_record
from conftest import tiny_backbone

QUESTIONS = {
    "color": {"type": "choice", "instructions": "Choose its color.",
              "criteria": {"blue": "blue", "red": "a much longer red option", "green": "green"}},
    "known": {"type": "noul", "instructions": "Is the color stated?"},
}
# The last chunk is a single token, which takes the recurrent single-step update path.
CHUNKS = ("The signal is blue.", " Later it turned red", "!")


@pytest.fixture(params=["none", "last-full-attention", "last-readout-cross"])
def session_model(request, tokenizer):
    torch.manual_seed(123)
    backbone = tiny_backbone(len(tokenizer), hybrid=True)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0,
                     weights_dtype="fp32", attention="eager", rows_per_forward=2,
                     candidate_interaction=request.param)
    model = QevModel(backbone, spec, tokenizer.pad_token_id)
    model.to(os.environ.get("QEV_TEST_DEVICE", "cpu")).eval()
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.fill_(0.8)
    return model


def one_shot(model, encoder, state, questions=QUESTIONS, **kwargs):
    encoded = encoder(typed_record({"state": state, "questions": questions}, source="inference",
                                   record_id="session"))
    return encoded, model.predict(encoded, **kwargs)


def probabilities(answer):
    return [torch.tensor(list(v["probabilities"].values()), dtype=torch.float32) for v in answer.values()]


def snapshot(cache):
    return [t.clone() for layer in cache.layers for t in vars(layer).values() if isinstance(t, torch.Tensor)] + [
        t.clone() for layer in cache.layers for d in vars(layer).values() if isinstance(d, dict)
        for t in d.values() if isinstance(t, torch.Tensor)]


def test_extensions_compute_only_new_state_tokens(session_model, encoder, monkeypatch):
    model = session_model
    calls = []
    original = model._run_rows

    def trace(rows, **kwargs):
        calls.append((rows, kwargs.get("offset", 0)))
        return original(rows, **kwargs)

    monkeypatch.setattr(model, "_run_rows", trace)
    session = Qev(model, encoder).session(CHUNKS[0])
    state = ""
    for chunk in CHUNKS:
        if state:
            session.extend(chunk)
        state += chunk
        before = session.cached_tokens
        encoded, reference = one_shot(model, encoder, state, cached=False)
        calls.clear()
        answer = session.predict(QUESTIONS)
        assert session.state == state and encoded.state == session._tokens
        assert calls[0] == ([encoded.state[before:]], before)
        assert sum(len(r) for rows, offset in calls if offset < len(encoded.state) for r in rows) == len(encoded.state) - before
        for got, expected in zip(probabilities(answer), reference):
            torch.testing.assert_close(got, expected.cpu().float(), atol=2e-5, rtol=2e-5)
        assert session.cached_tokens == len(encoded.state)
    calls.clear()
    session.predict(QUESTIONS)
    assert all(offset >= session.state_tokens for _, offset in calls)


def test_questions_leave_state_cache_unchanged(session_model, encoder):
    session = Qev(session_model, encoder).session(CHUNKS[0])
    first = session.predict(QUESTIONS)
    before = snapshot(session._cache)
    other = {"x": {"type": "score", "instructions": "Rate urgency.", "criteria": ["low", "mid", "high"]}}
    session.predict(other)
    for a, b in zip(before, snapshot(session._cache), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(probabilities(first), probabilities(session.predict(QUESTIONS)), atol=0, rtol=0)


@pytest.mark.parametrize("execution", ["reference", "tree"])
def test_session_executions_agree(session_model, encoder, execution):
    cached, other = Qev(session_model, encoder).session(), Qev(session_model, encoder, execution=execution).session()
    for chunk in CHUNKS:
        cached.extend(chunk)
        other.extend(chunk)
        torch.testing.assert_close(probabilities(cached.predict(QUESTIONS)), probabilities(other.predict(QUESTIONS)),
                                   atol=2e-5, rtol=2e-5)
    assert other.cached_tokens == 0


def test_overflow_and_failed_extension_preserve_session(session_model, encoder, monkeypatch):
    session = Qev(session_model, encoder).session(CHUNKS[0])
    session.predict(QUESTIONS)
    text, tokens = session.state, session.state_tokens
    with pytest.raises(ContextOverflow):
        session.extend("x" * encoder.limits.max_state)
    assert (session.state, session.state_tokens, session.cached_tokens) == (text, tokens, tokens)
    session.extend(CHUNKS[1])
    original = session_model._run_rows
    monkeypatch.setattr(session_model, "_run_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom")))
    with pytest.raises(RuntimeError, match="oom"):
        session.predict(QUESTIONS)
    assert session._cache is None and session.cached_tokens == 0
    monkeypatch.setattr(session_model, "_run_rows", original)
    _, reference = one_shot(session_model, encoder, CHUNKS[0] + CHUNKS[1], cached=False)
    for got, expected in zip(probabilities(session.predict(QUESTIONS)), reference):
        torch.testing.assert_close(got, expected.cpu().float(), atol=2e-5, rtol=2e-5)
    session.close()
    assert session._cache is None
