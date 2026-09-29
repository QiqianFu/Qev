# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
import math

import pytest
import torch

from qev.evaluate import checked_probabilities, summarize


def test_metric_denominators_and_calibration():
    rows = [
        {"source": "a", "correct": True, "confidence": 0.8, "brier": 0.08, "nll": -math.log(0.8)},
        {"source": "a", "correct": False, "confidence": 0.6, "brier": 0.72, "nll": -math.log(0.4)},
        {"source": "b", "correct": True, "confidence": 0.9, "brier": 0.02, "nll": -math.log(0.9)},
    ]
    r = summarize(rows, 2, {"b": 1, "c": 1})
    assert r["accuracy_answered"] == pytest.approx(2 / 3)
    assert r["accuracy_counting_rejections_wrong"] == pytest.approx(2 / 5)
    assert r["macro_accuracy_answered"] == 0.75
    assert r["macro_accuracy_counting_rejections_wrong"] == pytest.approx(1 / 3)
    assert r["brier"] == pytest.approx(0.82 / 3)
    assert r["nll"] == pytest.approx(-math.log(0.8 * 0.4 * 0.9) / 3)
    assert r["ece_10_bins"] == pytest.approx(0.3)


def test_invalid_predictions_fail_instead_of_becoming_class_zero():
    for values in [[float("nan"), 0.5], [-0.2, 1.2], [0.2, 0.2], [1.0]]:
        with pytest.raises(ValueError, match="probabilit"):
            checked_probabilities(torch.tensor(values), 2)
    assert checked_probabilities(torch.tensor([0.2, 0.8]), 2).tolist() == pytest.approx([0.2, 0.8])


def test_batched_cli_preserves_predictions_rejections_and_tail(tmp_path, monkeypatch, model, encoder, record):
    from dataclasses import replace
    import json
    import sys
    from qev import evaluate

    records = [replace(record, id=f"r{i}", state=("too long " * 100 if i == 1 else record.state)) for i in range(5)]
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(evaluate, "load_records", lambda *args: (records, {"files": {"dev": {"role": "development"}}}))
    monkeypatch.setattr(evaluate, "load_model", lambda *args, **kwargs: (model, None, encoder, {}))
    results = []
    for size in (1, 3):
        out = tmp_path / f"batch{size}"
        monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "unused", "--data", str(tmp_path),
                                         "--split", "dev", "--out", str(out), "--device", "cpu", "--reference",
                                         "--batch-size", str(size), "--progress-every", "0"])
        evaluate.main()
        rows = [json.loads(s) for s in (out / "predictions.jsonl").read_text().splitlines()]
        report = json.loads((out / "report.json").read_text())
        assert report["rejected_questions"] == 2
        assert report["answered_questions"] == 8
        assert report["inference_seconds"] > 0
        results.append(rows)
    for a, b in zip(*results, strict=True):
        for key in ("record_id", "question_id", "label", "prediction", "correct"):
            assert a[key] == b[key]
        assert a["probabilities"] == pytest.approx(b["probabilities"], abs=2e-6)
        assert b["timing_scope"] == "amortized_batch"
