# Adapted for Qev in 2026; see NOTICE and provenance.json.
from dataclasses import asdict, replace
import json

import pytest
import torch
from torch.nn import functional as F

from qev.checkpoint import load_model, save_model
from qev.encoding import Limits
from qev.model import QevModel, ModelSpec, record_loss
from qev.schema import Candidate, typed_record
from conftest import tiny_backbone


def joint_pair(tokenizer, interaction="last-full-attention"):
    """Plain and joint models sharing one hybrid backbone and one head."""
    torch.manual_seed(123)
    backbone = tiny_backbone(len(tokenizer), hybrid=True)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0,
                     weights_dtype="fp32", attention="eager", rows_per_forward=2)
    plain = QevModel(backbone, spec, tokenizer.pad_token_id).eval()
    joint = QevModel(backbone, replace(spec, candidate_interaction=interaction), tokenizer.pad_token_id).eval()
    joint.head.load_state_dict(plain.head.state_dict())
    return plain, joint


def set_gate(model, value):
    with torch.no_grad():
        model.joint_gate.fill_(value)


def test_zero_gate_equals_independent_branches(tokenizer, encoder, record):
    plain, joint = joint_pair(tokenizer)
    enc = encoder(record)
    for (q1, c1), (q2, c2) in zip(plain.reference_features(enc), joint.reference_features(enc)):
        torch.testing.assert_close(q1, q2, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(c1, c2, atol=2e-5, rtol=2e-5)
    for p1, p2 in zip(plain.predict(enc, cached=False), joint.predict(enc)):
        torch.testing.assert_close(p1, p2, atol=2e-6, rtol=2e-5)


def test_open_gate_is_permutation_equivariant(tokenizer, encoder, record):
    _, joint = joint_pair(tokenizer)
    set_gate(joint, 2.0)
    q = record.questions[0]
    perm = [2, 0, 1]
    reordered = replace(q, candidates=tuple(q.candidates[i] for i in perm), target=tuple(q.target[i] for i in perm))
    changed = replace(record, questions=(reordered, *record.questions[1:]))
    original = joint.predict(encoder(record))[0]
    torch.testing.assert_close(joint.predict(encoder(changed))[0], original[perm], atol=2e-6, rtol=2e-5)


def test_candidates_see_siblings_only_through_open_gate(tokenizer, encoder, record):
    _, joint = joint_pair(tokenizer)
    q = record.questions[0]
    changed_q = replace(q, candidates=(q.candidates[0], Candidate("red", "SECRET other much longer candidate text"), q.candidates[2]))
    changed = replace(record, questions=(changed_q, *record.questions[1:]))
    set_gate(joint, 0.0)
    base, mod = joint.reference_features(encoder(record)), joint.reference_features(encoder(changed))
    torch.testing.assert_close(base[0][1][0], mod[0][1][0], atol=2e-5, rtol=2e-5)
    set_gate(joint, 2.0)
    base, mod = joint.reference_features(encoder(record)), joint.reference_features(encoder(changed))
    assert (base[0][1][0] - mod[0][1][0]).abs().max() > 1e-4
    # The question readout and the sibling question never see these candidates.
    torch.testing.assert_close(base[0][0], mod[0][0], atol=2e-5, rtol=2e-5)
    for a, b in zip(base[1], mod[1]):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_open_gate_chunking_and_record_pooling_invariance(tokenizer, encoder, record):
    _, joint = joint_pair(tokenizer)
    set_gate(joint, 1.5)
    other = replace(record, id="r1", state="The signal is red and bright.")
    records = [encoder(record), encoder(other)]
    pooled = joint.predict_batch(records)
    joint.spec.rows_per_forward = 1
    for enc, expected in zip(records, pooled):
        for a, b in zip(joint.predict(enc), expected):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_gate_gradient_and_head_warmup(tokenizer, encoder, record):
    _, joint = joint_pair(tokenizer)
    joint.train()
    enc = encoder(record)
    record_loss(joint([enc])[0], enc).backward()
    # d/dg of tanh(g)*cross at g=0 is the cross term itself.
    assert joint.joint_gate.grad is not None and joint.joint_gate.grad.abs().sum() > 0
    assert joint.backbone.get_input_embeddings().weight.grad[enc.state[1]].abs().sum() > 0
    joint.zero_grad(set_to_none=True)
    joint.head_only = True
    record_loss(joint([enc])[0], enc).backward()
    assert joint.joint_gate.grad is None
    assert all(p.grad is None for p in joint.backbone.parameters())
    assert joint.head.project.weight.grad is not None


def test_non_gated_backbone_is_rejected(tokenizer):
    backbone = tiny_backbone(len(tokenizer), hybrid=False)
    spec = ModelSpec("tiny", head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype="fp32",
                     attention="eager", candidate_interaction="last-full-attention")
    with pytest.raises(NotImplementedError, match="gated"):
        QevModel(backbone, spec, tokenizer.pad_token_id)
    with pytest.raises(ValueError, match="candidate interaction"):
        replace(spec, candidate_interaction="all-layers")


def test_joint_checkpoint_round_trip(tokenizer, encoder, record, tmp_path):
    torch.manual_seed(0)
    base = tmp_path / "base"
    tiny_backbone(len(tokenizer), hybrid=True).save_pretrained(base)
    from qev.model import load_backbone
    spec = ModelSpec(str(base), head_dim=32, head_heads=4, head_layers=1, lora_rank=2, weights_dtype="fp32",
                     attention="eager", candidate_interaction="last-full-attention")
    model = QevModel(load_backbone(spec, "cpu"), spec, tokenizer.pad_token_id).eval()
    set_gate(model, 0.7)
    limits = Limits(128, 128, 128, 384, 32)
    save_model(tmp_path / "ckpt", model, tokenizer, limits)
    loaded, _, loaded_encoder, _ = load_model(tmp_path / "ckpt")
    torch.testing.assert_close(loaded.joint_gate, model.joint_gate)
    enc = loaded_encoder(record)
    for a, b in zip(model.predict(enc), loaded.eval().predict(enc)):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def score_record(levels=("bad", "ok", "good", "great"), label="2"):
    return typed_record({"state": "The meal was quite good.", "questions": {
        "rating": {"type": "score", "instructions": "Rate it.", "criteria": list(levels), "label": label}}},
        source="synthetic", record_id="s0", group_id="s0")


def test_rps_matches_definition_and_uses_level_ids(encoder):
    enc = encoder(score_record())
    z = torch.tensor([0.3, -1.0, 2.0, 0.5])
    p = z.softmax(-1)
    target = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rps = (p.cumsum(-1)[:-1] - target.cumsum(-1)[:-1]).square().mean()
    ce = -F.log_softmax(z, -1)[2]
    torch.testing.assert_close(record_loss([z], enc, ord_w=0.5), ce + 0.5 * rps)
    torch.testing.assert_close(record_loss([z], enc), ce)
    # Reordering candidate rows must not change the ordinal term.
    q = enc.questions[0]
    perm = [3, 1, 0, 2]
    question = replace(q.question, candidates=tuple(q.question.candidates[i] for i in perm),
                       target=tuple(q.question.target[i] for i in perm))
    shuffled = replace(enc, questions=(replace(q, question=question, candidates=tuple(q.candidates[i] for i in perm)),))
    torch.testing.assert_close(record_loss([z[perm]], shuffled, ord_w=0.5), ce + 0.5 * rps)


def test_hl_gauss_matches_definition_and_uses_level_ids(encoder):
    from math import erf, sqrt
    enc = encoder(score_record())
    z = torch.tensor([0.3, -1.0, 2.0, 0.5])
    sigma = 0.75
    cdf = lambda x: 0.5 * (1 + erf(x / (sigma * sqrt(2))))
    mass = torch.tensor([cdf(i + 0.5 - 2) - cdf(i - 0.5 - 2) for i in range(4)])
    target = mass / mass.sum()
    expected = -(target * F.log_softmax(z, -1)).sum()
    torch.testing.assert_close(record_loss([z], enc, score_hl_sigma=sigma), expected)
    torch.testing.assert_close(record_loss([z], enc, score_hl_sigma=0.0), record_loss([z], enc))
    q = enc.questions[0]
    perm = [3, 1, 0, 2]
    question = replace(q.question, candidates=tuple(q.question.candidates[i] for i in perm),
                       target=tuple(q.question.target[i] for i in perm))
    shuffled = replace(enc, questions=(replace(q, question=question, candidates=tuple(q.candidates[i] for i in perm)),))
    torch.testing.assert_close(record_loss([z[perm]], shuffled, score_hl_sigma=sigma), expected)


def test_hl_gauss_target_conserves_soft_mass_and_is_symmetric():
    from qev.model import hl_gauss_target
    soft = torch.tensor([0.1, 0.0, 0.6, 0.3])
    out = hl_gauss_target(soft, [0, 1, 2, 3], 0.75)
    torch.testing.assert_close(out.sum(), torch.tensor(1.0))
    assert (out > 0).all()
    centre = hl_gauss_target(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]), list(range(5)), 0.75)
    torch.testing.assert_close(centre[1], centre[3])
    assert centre[2] > centre[1] > centre[0]


def test_hl_gauss_ignores_choice_questions(encoder, record):
    enc = encoder(record)
    logits = [torch.randn(len(q.candidates)) for q in enc.questions]
    torch.testing.assert_close(record_loss(logits, enc, score_hl_sigma=0.75), record_loss(logits, enc))


def test_rps_ignores_choice_questions(encoder, record):
    enc = encoder(record)
    logits = [torch.randn(len(q.candidates)) for q in enc.questions]
    torch.testing.assert_close(record_loss(logits, enc, ord_w=3.0), record_loss(logits, enc))


def test_joint_rps_training_updates_gate_and_resumes_exactly(tmp_path, tokenizer, record):
    from safetensors.torch import load_file
    from qev.data import file_hash
    from test_training import run_training
    base = tmp_path / "base"
    torch.manual_seed(6)
    tiny_backbone(len(tokenizer), hybrid=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    rows = [replace(record, id=f"r{i}", group_id=f"g{i}") for i in range(2)]
    rows += [replace(score_record(label=str(i)), id=f"s{i}", group_id=f"s{i}") for i in range(2)]
    train = data / "train.jsonl"
    train.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    (data / "manifest.json").write_text(json.dumps({"files": {
        "train": {"file": "train.jsonl", "sha256": file_hash(train), "records": 4, "role": "train"}}}))
    spec = ModelSpec(str(base), head_dim=32, head_heads=4, head_layers=1, lora_rank=2, weights_dtype="fp32",
                     attention="eager", rows_per_forward=2, candidate_interaction="last-full-attention")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": asdict(spec), "limits": asdict(Limits(128, 128, 128, 384, 32)),
        "training": {"epochs": 2, "batch_size": 1, "accum": 2, "seed": 7, "lr": 0.001, "head_lr": 0.002,
                     "joint_gate_lr": 0.05, "ord_w": 0.5, "warmup_steps": 0, "head_warmup_steps": 1,
                     "gradient_checkpointing": True, "save_every": 1, "autocast": "fp32"}}))
    full, resumed = tmp_path / "full", tmp_path / "resume"
    run_training(data, config, full)
    run_training(data, config, resumed, "--max-steps", "2")
    run_training(data, config, resumed, "--resume", str(resumed / "step-000002"))
    gates = [load_file(str(full / f"step-{s:06d}" / "joint.safetensors"))["gate"] for s in (1, 4)]
    assert torch.equal(gates[0], torch.zeros_like(gates[0]))  # frozen during head warmup
    assert gates[1].abs().min() > 0
    for rel in ["head.safetensors", "joint.safetensors", "adapter/adapter_model.safetensors"]:
        a, b = load_file(str(full / "step-000004" / rel)), load_file(str(resumed / "step-000004" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    logs = [json.loads(l) for l in (full / "train.jsonl").read_text().splitlines()]
    assert logs[-1]["joint_gate_tanh"] is not None


def test_joint_gradients_match_with_and_without_checkpointing(tokenizer, encoder, record):
    from qev.execution import configure_checkpointing
    _, joint = joint_pair(tokenizer)
    set_gate(joint, 0.8)
    joint.train()
    enc = encoder(record)
    grads = []
    for enabled in (False, True):
        configure_checkpointing(joint.backbone, enabled)
        joint.zero_grad(set_to_none=True)
        record_loss(joint([enc])[0], enc).backward()
        grads.append({n: p.grad.clone() for n, p in joint.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and "joint_gate" in grads[0]
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("interaction", ["last-full-attention", "last-readout-cross"])
def test_readout_cross_reads_only_sibling_readouts(tokenizer, interaction):
    """With the gate open, perturbing a sibling's non-final residual at the joint layer input changes the other
    candidates only when the cross reads all sibling tokens; the readout variant reads each sibling's final token."""
    _, joint = joint_pair(tokenizer, interaction)
    set_gate(joint, 2.0)
    torch.manual_seed(5)
    hidden = joint.backbone.config.hidden_size
    prefix, widths = 3, [7, 6, 8]
    from qev.model import _base_model
    rot = _base_model(joint.backbone).rotary_emb
    rows = []
    for w in widths:
        h = torch.randn(w, hidden)
        cos, sin = rot(h[None], torch.arange(w)[None, None].expand(3, 1, w))
        rows.append((h, cos[0], sin[0]))
    _, base = joint._joint_features(rows, prefix)
    h1 = rows[1][0].clone()
    h1[prefix + 1] += 3.0  # a non-final token of candidate 1's span
    _, moved = joint._joint_features([rows[0], (h1, *rows[1][1:]), rows[2]], prefix)
    others = [0, 2]
    if interaction == "last-readout-cross":
        torch.testing.assert_close(moved[others], base[others], atol=1e-5, rtol=1e-5)
    else:
        assert (moved[others] - base[others]).abs().max() > 1e-4
    assert (moved[1] - base[1]).abs().max() > 1e-4  # candidate 1's own branch sees the edit
