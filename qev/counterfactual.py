# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Deterministic, label-blind edits and probability-response supervision.

Edits are proposals, not claims of semantic validity. A confident teacher is
also not a correctness oracle. All patches can be replayed on their exact input.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math
import re

from .distillation import question_key

SCHEMA = "qev.counterfactual.v1"
EDIT_RECIPE = "local-edits.v2-protected-task-sentences"
OPERATORS = ("delete_sentence", "number_change", "condition_change")
SKIP_KEYS = {"id", "speaker", "model", "source", "url", "date", "timestamp"}
NUMBER = re.compile(r"(?<![\w./:+-])\d+(?:\.\d+)?(?![\w/:+-]|\.\d)")
CONDITIONS = {
    "is less than or equal to": "is greater than", "is greater than or equal to": "is less than",
    "less than or equal to": "greater than", "greater than or equal to": "less than",
    "is not equal to": "is equal to", "is equal to": "is not equal to",
    "is at least": "is less than", "is less than": "is at least",
    "is greater than": "is at most", "is at most": "is greater than",
    "no later than": "later than", "no less than": "less than", "no more than": "more than", "at least": "less than",
    "at most": "more than", "greater than": "at most", "less than": "at least",
}
CONDITION = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(CONDITIONS, key=len, reverse=True))) + r")\b", re.I)
TASK_SENTENCE = re.compile(r"[?？]|\b(?:which|select|choose|answer|following|question|classify|identify|determine|evaluate)\b", re.I)


def deletable_observation(text):
    """Conservative lexical guard, not a semantic-validity classifier."""
    return bool(text.strip()) and TASK_SENTENCE.search(text) is None


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def blind_record(record, question):
    q = replace(question, target=None, label=None)
    key = question_key(record.state, q)
    return replace(record, id=key, questions=(q,))


def _parse(state):
    try:
        obj = json.loads(state)
        if isinstance(obj, (dict, list)):
            return obj, True
    except ValueError:
        pass
    return state, False


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            if key.lower() not in SKIP_KEYS:
                yield from _leaves(value[key], path + (key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _leaves(item, path + (i,))
    else:
        yield path, value


def replay(state, patch):
    if digest(state) != patch["state_sha256"]:
        raise ValueError("patch belongs to a different state")
    obj, structured = _parse(state)
    path = patch["path"]
    current = obj
    for key in path:
        current = current[key]
    if current != patch["before"]:
        raise ValueError("patch before-value mismatch")
    new = deepcopy(obj)
    if path:
        parent = new
        for key in path[:-1]:
            parent = parent[key]
        if patch["kind"] == "remove_item":
            if not isinstance(parent, list):
                raise ValueError("only list observations can be removed")
            del parent[path[-1]]
        else:
            parent[path[-1]] = patch["after"]
    else:
        new = patch["after"]
    result = json.dumps(new, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if structured else new
    if not result.strip() or result == state:
        raise ValueError("empty or ineffective patch")
    return result


def edits(state, *, seed=17, key="", per_operator=1):
    """At most N proposals per operator; never alter candidates or instructions.

    JSON text leaves are edited after parsing. A list of speaker/text objects
    also supports removing a complete observation. Dates and embedded IDs are
    excluded from the conservative numeric regex; this is not a full parser.
    """
    if per_operator < 1:
        raise ValueError("per_operator must be positive")
    obj, structured = _parse(state)
    proposals = {op: [] for op in OPERATORS}

    def add(op, path, before, after, kind="replace", **extra):
        patch = dict(operator=op, path=list(path), kind=kind, before=before, after=after,
                     state_sha256=digest(state), **extra)
        changed = replay(state, patch)
        proposals[op].append((patch, changed))

    if structured and isinstance(obj, list) and len(obj) > 1 and all(
            isinstance(v, dict) and isinstance(v.get("text"), str) for v in obj):
        for i, value in enumerate(obj):
            if deletable_observation(value["text"]):
                add("delete_sentence", (i,), value, None, "remove_item", unit="observation")
    for path, value in _leaves(obj):
        if isinstance(value, bool):
            add("condition_change", path, value, not value, unit="boolean")
        elif isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            add("number_change", path, value, value + 1, unit="number")
        elif isinstance(value, str):
            # Boundaries require punctuation + whitespace, or an actual newline;
            # decimal points are not sentence boundaries. Abbreviations remain
            # a documented limitation of this first deterministic proposal set.
            boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?。！？])\s+|\n+", value)] + [len(value)]
            spans = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if value[a:b].strip()]
            if len(spans) > 1:
                for start, end in spans:
                    after = (value[:start] + value[end:]).strip()
                    fragment = value[start:end].strip()
                    if len(after) >= 8 and fragment.endswith((".", "。", "!", "！")) and deletable_observation(fragment):
                        add("delete_sentence", path, value, after, unit="sentence", span=[start, end])
            for match in NUMBER.finditer(value):
                old = Decimal(match.group())
                unit = Decimal(1).scaleb(old.as_tuple().exponent)
                for new in (old - unit, old + unit):
                    if new < 0:
                        continue
                    after = value[:match.start()] + format(new, "f") + value[match.end():]
                    add("number_change", path, value, after, unit="numeric_span", span=list(match.span()))
            for match in CONDITION.finditer(value):
                new = CONDITIONS[match.group().lower()]
                if match.group()[0].isupper():
                    new = new[0].upper() + new[1:]
                after = value[:match.start()] + new + value[match.end():]
                add("condition_change", path, value, after, unit="condition_span", span=list(match.span()))
    seen = set()
    result = []
    for op in OPERATORS:
        ordered = sorted(proposals[op], key=lambda x: digest([seed, key, x[0]]))
        count = 0
        for patch, changed in ordered:
            if changed in seen:
                continue
            seen.add(changed)
            result.append((patch, changed))
            count += 1
            if count == per_operator:
                break
    return result


def response_stats(original, edited):
    if len(original) != len(edited) or not original:
        raise ValueError("response candidate counts differ")
    for ps in (original, edited):
        if any(not math.isfinite(p) or p < 0 for p in ps) or not math.isclose(sum(ps), 1., abs_tol=1e-6):
            raise ValueError("invalid response probabilities")
    delta = [b-a for a, b in zip(original, edited)]
    tv = sum(map(abs, delta))/2
    return {"delta": delta, "tv": tv,
            "flipped": max(range(len(original)), key=original.__getitem__) != max(range(len(edited)), key=edited.__getitem__),
            "bucket": "stable" if tv < .05 else "sensitive" if tv >= .2 else "intermediate"}
