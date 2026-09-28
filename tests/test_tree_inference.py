# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Packed hybrid trees must agree with full paths and keep all branch boundaries."""
from dataclasses import replace
import os

import pytest
import torch

from qev.encoding import packed_tree
from qev.model import QevModel, ModelSpec, lora_modules
from qev.schema import Candidate
from qev.tree import TreeLayout
from conftest import tiny_backbone


@pytest.fixture(params=[("none", "eager"), ("none", "sdpa"),
                        ("last-full-attention", "eager"), ("last-full-attention", "sdpa"),
                        ("last-readout-cross", "eager"), ("last-readout-cross", "sdpa")])
def tree_model(request, tokenizer):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(991)
    config = tiny_backbone(len(tokenizer), hybrid=True).config
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    interaction, backend = request.param
    config._attn_implementation = backend
    # Exercise distinct key/value head counts in the DeltaNet as in real 9B.
    config.linear_num_value_heads = 4
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1,
                     lora_rank=0, candidate_interaction=interaction, attention=backend)
    model = QevModel(Qwen3_5TextModel(config), spec, tokenizer.pad_token_id)
    model.to(os.environ.get("QEV_TEST_DEVICE", "cpu")).eval()
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.copy_(torch.linspace(-0.7, 0.8, 4, device=model.device))
    return model


def assert_features(a, b):
    for (qa, ca), (qb, cb) in zip(a, b, strict=True):
        torch.testing.assert_close(qa, qb, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(ca, cb, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("single", [False, True])
def test_tree_matches_full_paths_and_cached(tree_model, encoder, record, single):
    q = record.questions[0]
    q = replace(q, candidates=(q.candidates[0], Candidate("red", "a much longer red candidate"), q.candidates[2]))
    record = replace(record, questions=(q,) if single else (q, record.questions[1]))
    e = encoder(record)
    expected = tree_model.reference_features(e)
    assert_features(expected, tree_model.tree_features(e))
    assert_features(tree_model.cached_features(e), tree_model.tree_features(e))
    for a, b in zip(tree_model.predict(e, cached=False), tree_model.predict(e, tree=True), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("single", [False, True])
def test_tree_positions_and_mask_match_independent_oracle(encoder, record, single):
    if single:
        record = replace(record, questions=record.questions[:1])
    e = encoder(record)
    ids, pos, allow = packed_tree(e)
    layout = TreeLayout(e, "cpu", 8192)
    assert layout.ids[0].tolist() == ids
    assert layout.positions[0].tolist() == pos
    assert layout.allow.tolist() == allow


def test_tree_sibling_and_question_isolation_and_permutation(tree_model, encoder, record):
    if tree_model.joint_layer is not None:
        with torch.no_grad():
            tree_model.joint_gate.zero_()
    a = tree_model.tree_features(encoder(record))
    q = record.questions[0]
    changed = replace(q, candidates=(q.candidates[0], Candidate("red", "unrelated long secret"), q.candidates[2]))
    b = tree_model.tree_features(encoder(replace(record, questions=(changed, record.questions[1]))))
    torch.testing.assert_close(a[0][0], b[0][0], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(a[0][1][[0, 2]], b[0][1][[0, 2]], atol=3e-5, rtol=3e-5)
    assert_features(a[1:], b[1:])
    if tree_model.joint_layer is not None:
        with torch.no_grad():
            tree_model.joint_gate.fill_(0.9)
    original = tree_model.predict(encoder(record), tree=True)
    order = (2, 0, 1)
    permuted = replace(q, candidates=tuple(q.candidates[i] for i in order))
    result = tree_model.predict(encoder(replace(record, questions=(permuted, record.questions[1]))), tree=True)
    torch.testing.assert_close(original[0][list(order)], result[0], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(original[1], result[1], atol=3e-6, rtol=3e-5)


def test_tree_single_token_branches_and_unequal_parent_lengths(tree_model, encoder, record):
    e = encoder(record)
    q0 = replace(e.questions[0], prefix=e.questions[0].prefix[:1],
                 candidates=tuple(c[:1] for c in e.questions[0].candidates))
    e = replace(e, state=e.state[:1], questions=(q0, e.questions[1]))
    assert_features(tree_model.reference_features(e), tree_model.tree_features(e))


def test_tree_nonzero_lora_is_used(tree_model, encoder, record):
    from peft import LoraConfig, get_peft_model
    tree_model.backbone = get_peft_model(tree_model.backbone, LoraConfig(
        task_type="FEATURE_EXTRACTION", r=2, lora_alpha=4,
        target_modules=lora_modules(tree_model.backbone.config)))
    with torch.no_grad():
        for name, p in tree_model.backbone.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.03)
    tree_model.eval()
    e = encoder(record)
    assert_features(tree_model.reference_features(e), tree_model.tree_features(e))


def test_tree_projects_each_unique_token_once(tree_model, encoder, record, monkeypatch):
    e = encoder(record)
    count = len(e.state) + sum(len(q.prefix) + sum(map(len, q.candidates)) for q in e.questions)
    calls, handles = {}, []

    def track(name):
        def hook(module, args):
            calls.setdefault(name, []).append(args[0].shape)
        return hook

    for name, module in tree_model.backbone.named_modules():
        if name.endswith(("in_proj_qkv", "gate_proj")):
            handles.append(module.register_forward_pre_hook(track(name)))

    def forbidden(*args, **kwargs):
        raise AssertionError("tree must not dispatch segmented/full leaf backbone forwards")

    monkeypatch.setattr(tree_model, "_run_rows", forbidden)
    try:
        tree_model.predict(e, tree=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 6
    for name, shapes in calls.items():
        assert len(shapes) == 1
        rows = sum(1 + len(q.candidates) for q in e.questions)
        expected = rows if tree_model.joint_layer == 3 and name.startswith("layers.3.") else count
        assert shapes[0].numel() // tree_model.backbone.config.hidden_size == expected


def test_tree_explicit_limits_and_prediction_guard(tree_model, encoder, record):
    e = encoder(record)
    tree_model.train()
    with pytest.raises(RuntimeError, match="call eval"):
        tree_model.predict(e, tree=True)
    tree_model.eval()
    with pytest.raises(ValueError, match="mutually exclusive"):
        tree_model.predict(e, cached=False, tree=True)
    tree_model.spec.max_padded_tokens = 2
    with pytest.raises(ValueError, match="unique tokens"):
        tree_model.tree_features(e)
