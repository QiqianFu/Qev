# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Frozen teacher logits for the exact, possibly augmented, student input.

Cached logits are unscaled; teacher temperature is applied when reading them.
Replacing the target by a mixture of the original target and teacher probabilities is equivalent (up to teacher entropy) to
supervised CE plus forward KL. Labels never enter the cache key or teacher input.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from .augment import NoneInserter, none_absent_view

SCHEMA = "qev.teacher-logits.v1"


def question_key(state, question):
    value = {"state": state, "type": question.type, "instructions": question.instructions,
             "candidates": [{"id": c.id, "text": c.text} for c in question.candidates]}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def view_protocol(config):
    s = config["training"]
    return {"limits": config.get("limits", {}),
            "choice_none_policy": config["model"].get("choice_none_policy", "as-provided"),
            "seed": int(s.get("seed", 17)), "epochs": int(s.get("epochs", 2)),
            "none_absent_prob": float(s.get("none_absent_prob", 0)),
            "none_insert_prob": float(s.get("none_insert_prob", 0)),
            "none_insert_absent_frac": float(s.get("none_insert_absent_frac", .5)),
            "none_insert_exempt_sources": s.get("none_insert_exempt_sources", []),
            "late_split": s.get("late_split"),
            "late_fraction": float(s.get("late_fraction", .2)),
            "late_repeats": int(s.get("late_repeats", 1))}


def training_views(encoded, n_main, encoder, config):
    """All views that the trainer can encounter, independent of DDP partition.

Repeated late records and sampler padding reuse the same (seed, epoch, ID) view.
The trainer admits late data only in its final epoch.
"""
    p = view_protocol(config)
    inserter = NoneInserter(encoder, p["none_insert_prob"], p["none_insert_absent_frac"])
    for epoch in range(p["epochs"]):
        for row in encoded if epoch == p["epochs"] - 1 else encoded[:n_main]:
            view, _ = none_absent_view(row, prob=p["none_absent_prob"], seed=p["seed"], epoch=epoch)
            if not view.record.source.startswith(tuple(p["none_insert_exempt_sources"])):
                view, _, _ = inserter(view, seed=p["seed"], epoch=epoch)
            yield view


def probabilities(logits):
    if not logits or any(not math.isfinite(z) for z in logits):
        raise ValueError("teacher logits must be nonempty and finite")
    weights = [math.exp(z - max(logits)) for z in logits]
    mass = sum(weights)
    return tuple(w / mass for w in weights)


class TeacherCache:
    def __init__(self, path, *, protocol, weight=1.0, temperature=1.0):
        self.path, self.weight = Path(path), float(weight)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("teacher temperature must be finite and positive")
        if not 0 < self.weight <= 1:
            raise ValueError("teacher weight must be in (0, 1]")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        m = self.manifest
        if m.get("schema") not in {SCHEMA, "branchkev.teacher-logits.v1"} or m.get("complete") is not True or m.get("temperature") != 1:
            raise ValueError("incomplete or incompatible teacher cache")
        if m["view_protocol"] != protocol:
            raise ValueError("teacher cache input protocol mismatch")
        self.rows = {}
        for entry in m["shards"]:
            name = entry["file"]
            if Path(name).name != name:
                raise ValueError("teacher shard must be a local filename")
            p = self.path / name
            count = 0
            with p.open() as f:
                for line in f:
                    row = json.loads(line)
                    ids = tuple(row["candidate_ids"])
                    ps = probabilities([z / self.temperature for z in row["logits"]])
                    if len(ids) != len(ps) or len(set(ids)) != len(ids):
                        raise ValueError("teacher candidate/logit mismatch")
                    if row["key"] in self.rows:
                        raise ValueError("duplicate teacher input key")
                    self.rows[row["key"]] = (ids, ps)
                    count += 1
            if count != entry["questions"]:
                raise ValueError("teacher shard question count mismatch")
        if len(self.rows) != m["questions"]:
            raise ValueError("teacher cache question count mismatch")

    def targets(self, encoded):
        result = []
        for q in encoded.questions:
            key = question_key(encoded.record.state, q.question)
            if key not in self.rows:
                raise ValueError(f"missing teacher input: {encoded.record.id}/{q.question.id}")
            ids, ps = self.rows[key]
            if ids != tuple(c.id for c in q.question.candidates):
                raise ValueError("teacher candidate order mismatch")
            result.append(ps)
        return result

    def apply(self, encoded):
        qs = []
        for q, ps in zip(encoded.questions, self.targets(encoded), strict=True):
            if self.weight == 1:
                target = ps
            else:
                if q.question.target is None:
                    raise ValueError("mixed distillation requires original targets")
                target = tuple((1-self.weight)*g + self.weight*p
                               for g, p in zip(q.question.target, ps, strict=True))
            qs.append(replace(q, question=replace(q.question, target=target)))
        return replace(encoded, questions=tuple(qs))

    def verify_coverage(self, views):
        seen, count = set(), 0
        for view in views:
            self.targets(view)
            count += len(view.questions)
            seen.update(question_key(view.record.state, q.question) for q in view.questions)
        return {"visited_questions": count, "unique_inputs": len(seen),
                "cache_inputs": len(self.rows), "weight": self.weight, "temperature": self.temperature,
                "complete_coverage": True}
