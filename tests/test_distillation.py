# SPDX-License-Identifier: Apache-2.0
"""CPU/stdlib tests for teacher provenance, input identity and augmentation."""
from dataclasses import replace
import json
import math

import pytest

from qev.data import file_hash
from qev.distillation import TeacherCache, probabilities, question_key, training_views, view_protocol
from qev.encoding import EncodedQuestion, EncodedRecord, Limits
from qev.schema import Candidate, Question, Record


def example():
    q = Question("q", "choice", "Color?", (Candidate("a", "blue"), Candidate("b", "red")), (1., 0.), "a")
    r = Record("r", "g", "test", "It is blue.", (q,))
    return EncodedRecord(r, (1, 2), (EncodedQuestion(q, (3,), ((4,), (5,))),))


def cache_files(tmp_path, views, cfg, logits=None):
    tmp_path.mkdir(exist_ok=True)
    rows = {}
    for v in views:
        for eq in v.questions:
            q = eq.question
            key = question_key(v.record.state, q)
            rows[key] = {"key": key, "candidate_ids": [c.id for c in q.candidates],
                         "logits": logits or [float(i) for i in range(len(q.candidates))]}
    shard = tmp_path / "logits-00.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    manifest = {"schema": "qev.teacher-logits.v1", "complete": True, "temperature": 1,
                "data_manifest_sha256": "dataset", "view_protocol": view_protocol(cfg),
                "questions": len(rows), "teacher": {"checkpoint": "test-fixture"},
                "shards": [{"file": shard.name, "sha256": file_hash(shard), "questions": len(rows)}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return file_hash(tmp_path / "manifest.json")


def config():
    return {"model": {"choice_none_policy": "as-provided"}, "training": {"epochs": 2}}


def test_input_identity_excludes_labels_but_binds_order_text_and_context():
    v = example(); q = v.questions[0].question
    key = question_key(v.record.state, q)
    assert key == question_key(v.record.state, replace(q, label="b", target=(0., 1.), id="other"))
    assert key != question_key("It is red.", q)
    assert key != question_key(v.record.state, replace(q, candidates=tuple(reversed(q.candidates))))
    assert key != question_key(v.record.state, replace(q, instructions="Not its color?"))
    assert key != question_key(v.record.state, replace(q, candidates=(q.candidates[0], Candidate("b", "green"))))


def test_teacher_target_does_not_change_input_or_gold(tmp_path):
    v = example(); cfg = config()
    sha = cache_files(tmp_path, [v], cfg, [0., math.log(3)])
    cache = TeacherCache(tmp_path, protocol=view_protocol(cfg))
    new = cache.apply(v)
    assert new.questions[0].question.target == pytest.approx((.25, .75))
    assert new.questions[0].question.label == "a"
    assert new.questions[0].candidates == v.questions[0].candidates and new.state == v.state
    assert v.questions[0].question.target == (1., 0.)
    mixed = TeacherCache(tmp_path, protocol=view_protocol(cfg), weight=.5)
    assert mixed.apply(v).questions[0].question.target == pytest.approx((.625, .375))


def test_cache_fails_closed_for_missing_input_hash_and_protocol(tmp_path):
    v = example(); cfg = config(); sha = cache_files(tmp_path, [v], cfg)
    cache = TeacherCache(tmp_path, protocol=view_protocol(cfg))
    with pytest.raises(ValueError, match="missing teacher input"):
        cache.apply(replace(v, record=replace(v.record, state="different")))
    changed = config(); changed['training']['epochs'] = 3
    with pytest.raises(ValueError, match="protocol"):
        TeacherCache(tmp_path, protocol=view_protocol(changed))


def test_teacher_views_include_none_insertions_and_late_records(tmp_path):
    class Encoder:
        limits = Limits()
        def candidate_tokens(self, text):
            return tuple(text.encode())
    cfg = config()
    cfg["training"].update(none_insert_prob=.999999, none_insert_absent_frac=1., late_split="late_train")
    main = example()
    late = replace(main, record=replace(main.record, id="late", state="Late example"))
    views = list(training_views([main, late], 1, Encoder(), cfg))
    assert len(views) == 3  # main in both epochs; late only in final epoch
    assert [v.record.id for v in views] == ["r", "r", "late"]
    assert all("a" not in [c.id for c in v.questions[0].question.candidates] for v in views)
    sha = cache_files(tmp_path, views, cfg)
    cache = TeacherCache(tmp_path, protocol=view_protocol(cfg))
    assert cache.verify_coverage(views)["complete_coverage"]
    with pytest.raises(ValueError, match="missing teacher input"):
        cache.apply(main)  # original distribution cannot substitute for an augmented input


@pytest.mark.parametrize("values", [[], [float('nan'), 0], [0, float('inf')]])
def test_invalid_teacher_logits_rejected(values):
    with pytest.raises(ValueError):
        probabilities(values)


def test_softmax_handles_large_offset():
    assert probabilities([10000.,10000.]) == (.5,.5)
