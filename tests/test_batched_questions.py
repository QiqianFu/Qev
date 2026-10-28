# SPDX-License-Identifier: Apache-2.0
"""Questions over a state cache must run as packed trees and match per-question forks."""
from dataclasses import replace
import os

import pytest
import torch

from qev import Qev
from qev.model import QevModel, ModelSpec
from qev.schema import typed_record
import qev.tree
from conftest import tiny_backbone

DEVICE = os.environ.get("QEV_TEST_DEVICE", "cpu")
# Mixed types, ragged question prefixes and ragged candidates.
QUESTIONS = {
    "color": {"type": "choice", "instructions": "Choose its color.",
              "criteria": {"blue": "blue", "red": "a much longer red option", "green": "green",
                           "teal": "teal or another colour that is close to it", "none": "none"}},
    "known": {"type": "noul", "instructions": "Is the color stated anywhere in the long message above?"},
    "x": {"type": "noul", "instructions": "?"},
    "urgency": {"type": "score", "instructions": "Rate urgency.", "criteria": ["low", "mid", "high", "max"]},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {f"t{i}": "team " * (i % 4 + 1) + str(i) for i in range(9)}},
    "later": {"type": "noul", "instructions": "Does it turn red later?"},
}


@pytest.fixture(params=["none", "last-full-attention", "last-readout-cross"])
def batch_model(request, tokenizer):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(7)
    config = tiny_backbone(len(tokenizer), hybrid=True).config
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    config.linear_num_value_heads = 4  # distinct DeltaNet key/value head counts, as in 9B
    config._attn_implementation = "sdpa"
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype="fp32",
                     attention="sdpa", rows_per_forward=2, candidate_interaction=request.param)
    model = QevModel(Qwen3_5TextModel(config), spec, tokenizer.pad_token_id).to(DEVICE).eval()
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.copy_(torch.linspace(-0.9, 1.1, 4, device=model.device))
    return model


def encode(encoder, questions=QUESTIONS, state="The signal is blue, then it turns red. " * 3):
    return encoder(typed_record({"state": state, "questions": questions}, source="synthetic", record_id="batch"))


def sequential(model, monkeypatch, *args):
    with monkeypatch.context() as patch:
        patch.setattr(QevModel, "batches_questions", property(lambda self: False))
        return model.state_cache_features(*args)


def assert_features(a, b, atol=2e-5):
    for (qa, ca), (qb, cb) in zip(a, b, strict=True):
        torch.testing.assert_close(qa, qb, atol=atol, rtol=atol)
        torch.testing.assert_close(ca, cb, atol=atol, rtol=atol)


def tensors(cache):
    return [t for layer in cache.layers for value in vars(layer).values()
            for t in (value.values() if isinstance(value, dict) else [value]) if isinstance(t, torch.Tensor)]


def trace_traversals(monkeypatch):
    groups = []
    original = qev.tree.state_tree_features

    def trace(model, questions, *args):
        groups.append(list(questions))
        return original(model, questions, *args)

    monkeypatch.setattr(qev.tree, "state_tree_features", trace)
    return groups


@pytest.mark.parametrize("rows", [1, 2, 32])
def test_batched_questions_match_sequential_forks_and_reference(batch_model, encoder, monkeypatch, rows):
    model = batch_model
    model.spec.rows_per_forward = rows
    enc = encode(encoder)
    assert len({len(q.prefix) for q in enc.questions}) >= 5
    cache = model.extend_state_cache(None, enc.state, 0)
    expected, reference = sequential(model, monkeypatch, enc, cache), model.reference_features(enc)
    groups = trace_traversals(monkeypatch)
    forks = []
    monkeypatch.setattr(model, "fork_cache", lambda *a: forks.append(a))
    monkeypatch.setattr(model, "_run_rows", lambda *a, **k: pytest.fail("batched questions use no row forwards"))
    features = model.state_cache_features(enc, cache)
    assert len(groups) == 1 and not forks
    assert_features(expected, features)
    assert_features(reference, features)


def test_traversals_respect_token_budget_and_fall_back_per_question(batch_model, encoder, monkeypatch):
    model = batch_model
    enc = encode(encoder)
    cache = model.extend_state_cache(None, enc.state, 0)
    expected = sequential(model, monkeypatch, enc, cache)
    sizes = [len(q.prefix) + sum(map(len, q.candidates)) for q in enc.questions]
    groups, forks = trace_traversals(monkeypatch), []
    original = model.fork_cache

    def fork(parent, copies=1):
        forks.append(parent is cache)
        return original(parent, copies)

    monkeypatch.setattr(model, "fork_cache", fork)
    # Questions over the budget still need each candidate within it.
    floor = max(len(c) for q in enc.questions for c in q.candidates)
    budgets = sum(sizes), max(sizes), sorted(sizes)[-2], sorted(sizes)[len(sizes) // 2], max(min(sizes), floor)
    mixed = 0
    for budget in budgets:
        groups.clear()
        forks.clear()
        model.spec.max_padded_tokens = budget
        assert_features(expected, model.state_cache_features(enc, cache))
        packed = [q for group in groups for q in group]
        assert all(sum(len(q.prefix) + sum(map(len, q.candidates)) for q in group) <= budget for group in groups)
        # Packed questions keep their order; only questions over the budget get their own forks.
        assert packed == [q for q, size in zip(enc.questions, sizes) if size <= budget]
        assert sum(forks) == sum(size > budget for size in sizes)
        mixed += 0 < len(packed) < len(sizes)
    assert mixed >= 2


def test_questions_do_not_see_each_other_or_padding(batch_model, encoder):
    model = batch_model
    enc = encode(encoder)
    cache = model.extend_state_cache(None, enc.state, 0)
    together = model.state_cache_features(enc, cache)
    for i, q in enumerate(enc.questions):
        alone = model.state_cache_features(encode(encoder, {q.question.id: QUESTIONS[q.question.id]}), cache)
        assert_features([together[i]], alone, atol=1e-5)


def test_state_cache_stays_bit_identical(batch_model, encoder):
    model = batch_model
    enc = encode(encoder)
    cache = model.extend_state_cache(None, enc.state, 0)
    before = [t.clone() for t in tensors(cache)]
    first = model.state_cache_features(enc, cache)
    for a, b in zip(before, tensors(cache), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert_features(first, model.state_cache_features(enc, cache), atol=0)


def test_many_choice_candidates_with_short_questions(batch_model, encoder, monkeypatch):
    model = batch_model
    model.spec.rows_per_forward = 3
    questions = {f"c{i}": {"type": "choice", "instructions": "Pick." * (i + 1),
                           "criteria": {f"o{j}": "option " * (j % 3 + 1) + str(j) for j in range(7 + i)}}
                 for i in range(4)}
    questions.update({f"n{i}": {"type": "noul", "instructions": "Yes?" * (i + 1)} for i in range(5)})
    questions["s"] = QUESTIONS["urgency"]
    enc = encode(encoder, questions)
    cache = model.extend_state_cache(None, enc.state, 0)
    expected = sequential(model, monkeypatch, enc, cache)
    sizes = [len(q.prefix) + sum(map(len, q.candidates)) for q in enc.questions]
    model.spec.max_padded_tokens = sum(sizes) // 3
    groups = trace_traversals(monkeypatch)
    assert_features(expected, model.state_cache_features(enc, cache))
    assert len(groups) >= 3 and sum(map(len, groups)) == len(enc.questions)


def test_layout_rows_positions_and_state_continuation(batch_model, encoder):
    model = batch_model
    enc = encode(encoder)
    ls = len(enc.state)
    layout = qev.tree.TreeLayout.over_state(enc.questions, ls, model.device, 10_000)
    nodes = layout.nodes
    roots = [n for n in nodes if n.parent < 0]
    assert len(roots) == len(enc.questions)
    positions = layout.positions[0].tolist()
    for root, q in zip(roots, enc.questions, strict=True):
        assert positions[root.start:root.end] == list(range(ls, ls + len(q.prefix)))
        children = [n for n in nodes if n.parent >= 0 and nodes[n.parent] is root]
        assert [layout.ids[0, c.start:c.end].tolist() for c in children] == [list(c) for c in q.candidates]
        for c in children:
            assert positions[c.start:c.end] == list(range(ls + len(q.prefix), ls + len(q.prefix) + c.end - c.start))
    total, kernel = layout.ids.numel(), 4
    for group, gather, history in layout.groups(kernel, 2, 10_000):
        for row, i in enumerate(group):
            n = nodes[i]
            path = list(range(total + 1, total + 1 + kernel)) + list(nodes[n.parent].path if n.parent >= 0 else ())
            assert history[row].tolist() == path[-kernel:]
            assert gather[row, :n.end - n.start].tolist() == list(range(n.start, n.end))
            assert (gather[row, n.end - n.start:] == total).all()
    # Question prefixes are padded together (identity DeltaNet steps); candidates follow in order.
    depth = [nodes[group[0]].depth for group, _ in layout.node_groups(32, 10_000)]
    assert depth == [0, 1]
    runs = [r for *_, r in layout.state_groups(2, 10_000) if r]
    for (*_, group_runs), (group, _) in zip(layout.state_groups(2, 10_000), layout.node_groups(2, 10_000)):
        for start, end, first, last in group_runs:
            assert {(nodes[nodes[i].parent].start, nodes[nodes[i].parent].end) for i in group[first:last]} == {(start, end)}
    assert runs


def test_sessions_batch_questions(batch_model, encoder, monkeypatch):
    model = batch_model
    session = Qev(model, encoder).session("The signal is blue.")
    session.extend(" Later it turned red.")
    groups = trace_traversals(monkeypatch)
    answer = session.predict(QUESTIONS)
    assert len(groups) == 1 and len(groups[0]) == len(QUESTIONS)
    reference = Qev(model, encoder, execution="reference").session("The signal is blue.")
    reference.extend(" Later it turned red.")
    expected = reference.predict(QUESTIONS)
    for key, value in expected.items():
        got = torch.tensor(list(answer[key]["probabilities"].values()))
        torch.testing.assert_close(got, torch.tensor(list(value["probabilities"].values())), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("state", ["", "ab"])
def test_short_state_and_prefixes_continue_the_convolution_window(batch_model, encoder, monkeypatch, state):
    model = batch_model
    kernel = model.backbone.config.linear_conv_kernel_dim
    enc = encode(encoder, state=state)
    assert len(enc.state) < kernel or state == "ab"
    # Prefixes shorter than the window: candidates also read the state's convolution inputs.
    short = tuple(replace(q, prefix=q.prefix[-n:]) for q, n in zip(enc.questions, (1, 2, 3, 5, 1, 2)))
    enc = replace(enc, questions=short)
    cache = model.extend_state_cache(None, enc.state, 0)
    assert_features(sequential(model, monkeypatch, enc, cache), model.state_cache_features(enc, cache))
    assert_features(model.reference_features(enc), model.state_cache_features(enc, cache))


def test_unaudited_layouts_use_sequential_forks(tokenizer, encoder, record, monkeypatch, batch_model):
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype="fp32",
                     attention="eager", rows_per_forward=2)
    plain = QevModel(tiny_backbone(len(tokenizer)), spec, tokenizer.pad_token_id).to(DEVICE).eval()
    assert batch_model.batches_questions and not plain.batches_questions
    enc = encoder(record)
    groups = trace_traversals(monkeypatch)
    features = plain.state_cache_features(enc, plain.extend_state_cache(None, enc.state, 0))
    assert_features(plain.reference_features(enc), features)
    monkeypatch.setattr(QevModel, "shares_prefix", property(lambda self: False))
    assert not batch_model.batches_questions
    features = batch_model.state_cache_features(enc, batch_model.extend_state_cache(None, enc.state, 0))
    assert_features(batch_model.reference_features(enc), features)
    assert not groups
