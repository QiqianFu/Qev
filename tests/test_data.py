# Adapted for Qev in 2026; see NOTICE and provenance.json.
import json
from pathlib import Path

import pytest

from qev.data import file_hash, load_records
from qev.adapters import from_typed, source_question
from qev.schema import Question, Candidate, typed_record


def test_soft_targets_preserved_and_not_leaked():
    row = {"id": "r", "workflow": "test", "state": json.dumps({"text": "observed"}),
           "questions": json.dumps({"q": {"type": "choice", "instructions": "Classify", "criteria": {"a": "A", "b": "B"}}}),
           "gold": json.dumps({"q": {"label": "b", "probabilities": {"a": 0.3, "b": 0.7}}}),
           "factors": "ANSWER LEAK"}
    r = from_typed(row)
    assert r.questions[0].target == (0.3, 0.7)
    assert "ANSWER LEAK" not in r.state
    assert "probabilities" not in r.questions[0].instructions


def test_training_refuses_eval_and_corrupt_hash(tmp_path, record):
    file = tmp_path / "eval.jsonl"
    file.write_text(json.dumps(record.to_dict()) + "\n")
    spec = {"files": {"eval": {"role": "test", "file": file.name, "sha256": file_hash(file), "records": 1}}}
    (tmp_path / "manifest.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="cannot be used"):
        load_records(tmp_path, "eval", training=True)
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_records(tmp_path, "eval")


def test_candidate_target_validation():
    with pytest.raises(ValueError, match="target distribution"):
        Question("q", "choice", "?", (Candidate("a", "A"),), (float("nan"),), "a")


def test_source_rendering_keeps_hypothesis_and_question():
    r = source_question("nyu-mll/multi_nli", {"premise": "P", "hypothesis": "H", "label": 1})
    assert r["state"] == {"premise": "P", "hypothesis": "H"}
    b = source_question("google/boolq", {"passage": "P", "question": "Q", "answer": False})
    assert b["state"] == "P" and b["questions"]["decision"]["instructions"] == "Q"


def test_long_input_is_rejected_not_truncated(encoder, record):
    from dataclasses import replace
    from qev.encoding import ContextOverflow
    with pytest.raises(ContextOverflow):
        encoder(replace(record, state="x" * 5000))
