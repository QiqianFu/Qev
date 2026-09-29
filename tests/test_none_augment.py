# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import replace
import json

import torch
from safetensors.torch import load_file

from qev.augment import none_absent_view
from qev.choice_policy import NONE_VARIANTS, ensure_choice_none, none_candidate
from qev.encoding import Encoder, leaf_rows
from qev.schema import Candidate, typed_record


def choice_record(i, criteria=None, label="a"):
    criteria = criteria or {"a": "alpha", "b": "beta", "c": "gamma"}
    return typed_record({"state": f"Case {i}.", "questions": {
        "pick": {"type": "choice", "instructions": "Pick one.", "criteria": criteria, "label": label},
        "rate": {"type": "score", "instructions": "Rate.", "criteria": ["x", "y", "z"], "label": "1"}}},
        source="s", record_id=f"r{i}", group_id=f"g{i}")


def test_varied_none_wording_is_label_free_deterministic_and_recognised(encoder):
    varied = Encoder(encoder.tokenizer, encoder.limits, choice_none_policy="always-varied")
    seen = set()
    for i in range(60):
        record = choice_record(i)
        updated = ensure_choice_none(record, varied=True)
        none = none_candidate(updated.questions[0])
        assert none is not None and none == updated.questions[0].candidates[-1]
        assert ensure_choice_none(updated, varied=True) == updated
        assert updated.questions[0].target[-1] == 0.0
        seen.add(none.text)
        relabelled = choice_record(i, label="c")
        assert leaf_rows(varied(record)) == leaf_rows(varied(relabelled))
    assert len(seen) == len(NONE_VARIANTS)


def test_varied_none_skips_ids_already_used_by_real_options():
    # "other" is a None ID only when its text says so; a real class may use it.
    criteria = {"a": "alpha", "other": "a different topic"}
    used = set()
    for i in range(60):
        q = ensure_choice_none(choice_record(i, criteria=criteria), varied=True).questions[0]
        assert len({c.id for c in q.candidates}) == len(q.candidates)
        none = none_candidate(q)
        assert none.id != "other" and q.candidates[1].text == "other: a different topic"
        used.add(none.text)
    assert len(used) == len(NONE_VARIANTS) - 1


def test_none_absent_view_relabels_consistently(encoder):
    varied = Encoder(encoder.tokenizer, encoder.limits, choice_none_policy="always-varied")
    enc = varied(choice_record(3))
    same, count = none_absent_view(enc, prob=0.0, seed=1, epoch=0)
    assert same is enc and count == 0
    view, count = none_absent_view(enc, prob=0.999999, seed=1, epoch=0)
    assert count == 1
    q, original = view.questions[0], enc.questions[0]
    none = none_candidate(original.question)
    assert [c.id for c in q.question.candidates] == [c.id for c in original.question.candidates if c.id != "a"]
    assert q.question.label == none.id and q.question.target[q.question.candidates.index(none)] == 1.0
    assert sum(q.question.target) == 1.0 and len(q.candidates) == len(q.question.target)
    # Token rows stay attached to their candidates; the prefix and Score question are untouched.
    kept = [i for i, c in enumerate(original.question.candidates) if c.id != "a"]
    assert q.candidates == tuple(original.candidates[i] for i in kept) and q.prefix == original.prefix
    assert view.questions[1] == enc.questions[1] and view.state == enc.state
    # Deterministic per (seed, epoch, record); a None-gold or soft question is never changed.
    assert none_absent_view(enc, prob=0.5, seed=1, epoch=4) == none_absent_view(enc, prob=0.5, seed=1, epoch=4)
    none_gold = replace(original.question, target=tuple(float(c == none) for c in original.question.candidates), label=none.id)
    soft = replace(original.question, target=(0.5, 0.5, 0.0, 0.0), label=None)
    for question in (none_gold, soft):
        record = replace(enc, questions=(replace(original, question=question), enc.questions[1]))
        assert none_absent_view(record, prob=0.999999, seed=1, epoch=0) == (record, 0)


def test_none_absent_rate_is_close_to_requested(encoder):
    varied = Encoder(encoder.tokenizer, encoder.limits, choice_none_policy="always-varied")
    encoded = [varied(choice_record(i)) for i in range(400)]
    count = sum(none_absent_view(e, prob=0.2, seed=17, epoch=0)[1] for e in encoded)
    assert 50 <= count <= 110


def test_training_with_varied_none_and_relabelling_resumes_exactly(tmp_path, tokenizer, record):
    from test_training import run_training, training_assets
    data, config = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(config.read_text())
    cfg["model"]["choice_none_policy"] = "always-varied"
    cfg["training"]["none_absent_prob"] = 0.5
    config.write_text(json.dumps(cfg))
    full, resumed = tmp_path / "full", tmp_path / "resume"
    run_training(data, config, full)
    run_training(data, config, resumed, "--max-steps", "1")
    run_training(data, config, resumed, "--resume", str(resumed / "step-000001"))
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        a, b = load_file(str(full / "step-000004" / rel)), load_file(str(resumed / "step-000004" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    logs = [json.loads(l) for l in (full / "train.jsonl").read_text().splitlines()]
    assert sum(l["none_absent_views_global"] for l in logs) > 0


def test_none_inserter_present_and_absent_views(encoder):
    from qev.augment import NoneInserter
    inserter = NoneInserter(encoder, 0.999999, 0.5)
    encoded = [encoder(choice_record(i)) for i in range(80)]
    kinds = {"present": 0, "absent": 0}
    for enc in encoded:
        view, added, absent = inserter(enc, seed=3, epoch=0)
        assert added == 1 and view.questions[1] == enc.questions[1] and view.state == enc.state
        q, original = view.questions[0], enc.questions[0]
        none = none_candidate(q.question)
        assert none is not None and none_candidate(original.question) is None
        assert len(q.candidates) == len(q.question.candidates) == len(q.question.target)
        assert q.candidates[q.question.candidates.index(none)] == encoder.candidate_tokens(none.text)
        assert q.prefix == original.prefix and sum(q.question.target) == 1.0
        if absent:
            kinds["absent"] += 1
            assert "a" not in {c.id for c in q.question.candidates} and q.question.label == none.id
        else:
            kinds["present"] += 1
            assert q.question.label == "a" and q.question.target[q.question.candidates.index(none)] == 0.0
            assert q.candidates[:-1] == original.candidates
        assert inserter(enc, seed=3, epoch=0) == (view, added, absent)
    assert kinds["absent"] > 20 and kinds["present"] > 20


def test_none_inserter_skips_existing_none_and_respects_zero_rate(encoder):
    from qev.augment import NoneInserter
    enc = encoder(ensure_choice_none(choice_record(1)))
    assert NoneInserter(encoder, 0.999999, 1.0)(enc, seed=1, epoch=0) == (enc, 0, 0)
    plain = encoder(choice_record(2))
    assert NoneInserter(encoder, 0.0, 1.0)(plain, seed=1, epoch=0) == (plain, 0, 0)


def test_training_with_none_insertion_resumes_exactly(tmp_path, tokenizer, record):
    from test_training import run_training, training_assets
    data, config = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(config.read_text())
    cfg["training"].update(none_insert_prob=0.9, none_insert_absent_frac=0.5)
    config.write_text(json.dumps(cfg))
    full, resumed = tmp_path / "full", tmp_path / "resume"
    run_training(data, config, full)
    run_training(data, config, resumed, "--max-steps", "1")
    run_training(data, config, resumed, "--resume", str(resumed / "step-000001"))
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        a, b = load_file(str(full / "step-000004" / rel)), load_file(str(resumed / "step-000004" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    logs = [json.loads(l) for l in (full / "train.jsonl").read_text().splitlines()]
    assert sum(l["none_inserted_global"] for l in logs) > 0
