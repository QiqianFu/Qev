# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Validated, target-independent input schema shared by preparation and inference."""
from dataclasses import asdict, dataclass
import json
import math

SCHEMA = "qev.record.v1"


def render(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    candidates: tuple[Candidate, ...]
    target: tuple[float, ...] | None = None
    label: str | None = None

    def __post_init__(self):
        if self.type not in {"choice", "score", "noul"}:
            raise ValueError(f"unsupported question type: {self.type}")
        ids = [c.id for c in self.candidates]
        if not 1 <= len(ids) <= 255 or len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique, with 1..255 candidates")
        if self.type == "noul" and set(ids) != {"false", "true"}:
            raise ValueError("noul requires false/true candidates")
        if self.label is not None and self.label not in ids:
            raise ValueError(f"label {self.label!r} is not a candidate")
        if self.target is not None:
            if len(self.target) != len(ids) or any(not math.isfinite(p) or p < 0 for p in self.target):
                raise ValueError("invalid target distribution")
            if not math.isclose(sum(self.target), 1.0, abs_tol=1e-6):
                raise ValueError("target distribution must sum to one")


@dataclass(frozen=True)
class Record:
    id: str
    group_id: str
    source: str
    state: str
    questions: tuple[Question, ...]

    def __post_init__(self):
        if not self.questions or len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("record requires nonempty, uniquely identified questions")

    def to_dict(self):
        return {"schema": SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") not in {SCHEMA, "branchkev.record.v1"}:
            raise ValueError("expected canonical qev.record.v1")
        qs = []
        for q in data["questions"]:
            qs.append(Question(q["id"], q["type"], q["instructions"],
                               tuple(Candidate(**c) for c in q["candidates"]),
                               tuple(q["target"]) if q.get("target") is not None else None,
                               q.get("label")))
        return cls(data["id"], data["group_id"], data["source"], data["state"], tuple(qs))


def normalize_target(values):
    values = [float(x) for x in values]
    if not values or any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("target probabilities must be finite and nonnegative")
    total = sum(values)
    if abs(total - 1) > 0.02:
        raise ValueError(f"target mass {total} is not rounding error")
    return tuple(v / total for v in values)


def typed_question(qid, q, gold=None):
    """Convert TypeSafe-shaped input; never expose labels or rationales in text."""
    typ = q["type"]
    criteria = q.get("criteria")
    if typ == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be a mapping")
        candidates = tuple(Candidate(str(k), str(k) if v in (None, "") else f"{k}: {render(v)}")
                           for k, v in criteria.items())
    elif typ == "noul":
        criteria = criteria or {}
        candidates = tuple(Candidate(k, k + (": " + render(criteria[k]) if k in criteria else ""))
                           for k in ("false", "true"))
    elif typ == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be a list")
        candidates = tuple(Candidate(str(i), f"level {i}: {render(v)}") for i, v in enumerate(criteria))
    else:
        raise ValueError(f"unsupported type {typ}")
    ids = [c.id for c in candidates]
    gold = gold or {}
    label = gold.get("label", q.get("label"))
    if isinstance(label, bool):
        label = str(label).lower()
    elif label is not None:
        label = str(label)
    target = gold.get("probabilities", q.get("target"))
    if target is not None:
        if isinstance(target, dict):
            if set(target) != set(ids):
                raise ValueError("target keys differ from candidate IDs")
            target = [target[k] for k in ids]
        target = normalize_target(target)
    elif label is not None:
        if label not in ids:
            raise ValueError(f"unknown label {label}")
        target = tuple(float(k == label) for k in ids)
    return Question(str(qid), typ, render(q.get("instructions")), candidates, target, label)


def typed_record(data, *, source, record_id=None, group_id=None, gold=None):
    meta = data.get("_meta", {})
    rid = str(record_id or meta.get("id") or data.get("id") or "")
    if not rid:
        raise ValueError("stable record ID is required")
    group = str(group_id or meta.get("group_id") or rid)
    return Record(rid, group, source, render(data["state"]),
                  tuple(typed_question(qid, q, (gold or {}).get(qid)) for qid, q in data["questions"].items()))
