# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from qev.counterfactual import blind_record, digest, edits, replay
from qev.data import file_hash, write_json
from qev.distillation import question_key
from qev.schema import Candidate, Question, Record


def sample(state="The value is 9. Approval requires at least 10. A clerk is present."):
    return Record("r", "group", "test", state, (Question("q", "noul", "Approve?",
        (Candidate("false", "false"), Candidate("true", "true")), (1., 0.), "false"),))


def test_patch_replay_is_deterministic_and_input_bound():
    state = sample().state
    pairs = edits(state, seed=3)
    assert {p["operator"] for p, _ in pairs} == {"delete_sentence", "number_change", "condition_change"}
    assert pairs == edits(state, seed=3)
    for patch, changed in pairs:
        assert replay(state, patch) == changed and changed != state
        with pytest.raises(ValueError, match="different state"):
            replay(state + " altered", patch)


def test_structured_observation_removal_preserves_json():
    state = json.dumps([{"speaker": "A", "text": "Value is 8."}, {"speaker": "B", "text": "At least 9 required."}])
    pairs = edits(state)
    removed = next(s for p, s in pairs if p["kind"] == "remove_item")
    assert len(json.loads(removed)) == 1
    for patch, changed in pairs:
        json.loads(changed)
        assert replay(state, patch) == changed
        assert "speaker" not in patch["path"]


def test_decimal_date_identifier_and_single_sentence():
    state = "ID AB-123 on 2027-04-15 has value 3.5."
    pairs = edits(state, per_operator=20)
    assert len(pairs) == 2
    assert all(p["operator"] == "number_change" for p, _ in pairs)
    assert all("AB-123" in s and "2027-04-15" in s for _, s in pairs)
    assert {s.rsplit(" ", 1)[-1] for _, s in pairs} == {"3.4.", "3.6."}


def test_labels_and_targets_do_not_change_edit_selection():
    original = sample()
    a = blind_record(original, original.questions[0])
    b = blind_record(original, replace(original.questions[0], label="true", target=(0., 1.)))
    assert a == b
    assert a.questions[0].label is None and a.questions[0].target is None
    assert edits(a.state, key=a.id) == edits(b.state, key=b.id)


def test_deletion_preserves_explicit_task_definition_and_question():
    bad = "One of the following sentences is nonsensical. Which one is it?"
    assert edits(bad) == []
    state = "The clerk is present. Which account should be chosen?"
    deleted = [s for p, s in edits(state) if p["operator"] == "delete_sentence"]
    assert deleted == ["Which account should be chosen?"]
    # Code-like newline fragments are not complete natural-language sentences.
    assert not [p for p, _ in edits("if enabled:\n    execute()\n    return True") if p["operator"] == "delete_sentence"]


@pytest.mark.parametrize("before,after", [("less than or equal to", "greater than"),
    ("greater than or equal to", "less than"), ("no more than", "more than")])
def test_compound_comparison_is_replaced_as_one_condition(before, after):
    state = f"Is 62 {before} 64?"
    changed = [s for p, s in edits(state) if p["operator"] == "condition_change"]
    assert changed == [f"Is 62 {after} 64?"]
