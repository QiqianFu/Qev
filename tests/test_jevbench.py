# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Semantic regressions at the JevBench/Qev format boundary."""
from copy import deepcopy

import pytest

from scripts.prepare_jevbench import convert_task, input_keys


def task(typ, labels, criteria, expected):
    return {"id": "fixture-1", "group": "paired-fixture", "family": "policy",
            "split": "public", "state": {"facts": "A request is pending."},
            "question": {"type": typ, "instructions": "Apply the rubric.", "criteria": criteria},
            "labels": labels, "expected": expected,
            "provenance": {"license": "MIT", "rationale": "SECRET GOLD EXPLANATION"}}


def test_native_choice_order_and_gold_are_independent():
    raw = task("choice", ["deny", "allow"], {"allow": "Allowed", "deny": "Denied"}, "deny")
    raw["provenance"]["gold_probs"] = {"deny": 0.75, "allow": 0.25}
    record, meta = convert_task(raw, "hard")
    q = record.questions[0]
    assert [c.id for c in q.candidates] == ["allow", "deny"]
    assert q.target == (0.25, 0.75) and q.label == "deny"
    assert record.group_id == "jevbench:paired-fixture"
    changed = deepcopy(raw)
    changed["expected"] = "allow"
    changed["provenance"]["gold_probs"] = {"deny": 0.1, "allow": 0.9}
    assert input_keys(record) == input_keys(convert_task(changed, "hard")[0])
    assert "SECRET" not in record.state + q.instructions + "".join(c.text for c in q.candidates)
    assert meta["provenance"] == raw["provenance"]


def test_noul_probability_and_label_mapping():
    raw = task("noul", ["no", "yes"], {"true": "Satisfied", "false": "Not satisfied"}, "yes")
    raw["provenance"]["gold_probs"] = {"no": 0.3818, "yes": 0.6182}
    record, meta = convert_task(raw, "hard")
    q = record.questions[0]
    assert [(c.id, c.text) for c in q.candidates] == [
        ("false", "false: Not satisfied"), ("true", "true: Satisfied")]
    assert q.label == "true" and q.target == (0.3818, 0.6182)
    assert meta["label_mapping"] == {"no": "false", "yes": "true"}


def test_score_retains_zero_based_ordinal_levels():
    raw = task("score", ["0", "1", "2"], ["None", "Some", "Many"], 2)
    record, _ = convert_task(raw, "original")
    q = record.questions[0]
    assert q.label == "2" and q.target == (0.0, 0.0, 1.0)
    assert [c.text for c in q.candidates] == ["level 0: None", "level 1: Some", "level 2: Many"]


def test_rejects_missing_gold_probability_key():
    raw = task("noul", ["no", "yes"], None, "no")
    raw["provenance"]["gold_probs"] = {"no": 1.0}
    with pytest.raises(ValueError, match="probability keys"):
        convert_task(raw, "hard")


def test_does_not_silently_renormalize_upstream_gold():
    raw = task("noul", ["no", "yes"], None, "yes")
    raw["provenance"]["gold_probs"] = {"no": 0.3, "yes": 0.701}
    with pytest.raises(ValueError, match="normalization"):
        convert_task(raw, "hard")


def test_prepared_data_verification_checks_tasks_roles_and_licenses(tmp_path, monkeypatch):
    import json
    import scripts.prepare_jevbench as prepare
    record, meta=convert_task(task('noul',['no','yes'],None,'yes'),'original')
    source=tmp_path/'source';source.mkdir()
    data=tmp_path/'model_data'/prepare.DATA_NAME;data.mkdir(parents=True)
    for name in ('LICENSE','THIRD-PARTY.md'):
        (source/name).write_text('Original upstream terms')
        (data/name).write_text('Original upstream terms')
    (data/'public.jsonl').write_text(json.dumps(record.to_dict())+'\n')
    (data/'provenance.jsonl').write_text(json.dumps(meta)+'\n')
    manifest={'revision':prepare.REVISION,'files':{'public':{
        'file':'public.jsonl','records':1,'role':'external_evaluation'}}}
    (data/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare,'fetch_source',lambda *args,**kwargs:source)
    monkeypatch.setattr(prepare,'read_source',lambda *args:({'public':[record]},[meta],{}))
    assert prepare.verify(tmp_path)['splits']=={'public':1}
    (data/'THIRD-PARTY.md').write_text('Changed terms')
    with pytest.raises(ValueError,match='THIRD-PARTY'):
        prepare.verify(tmp_path)
