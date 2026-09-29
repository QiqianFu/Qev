# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Training-only views of encoded records; evaluation never calls this module."""
from dataclasses import replace
import hashlib
import random

from .choice_policy import NONE_VARIANTS, none_candidate
from .schema import Candidate


def _rng(seed, epoch, record_id):
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{record_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def none_absent_view(encoded, *, prob, seed, epoch):
    """With probability prob per eligible Choice question, delete the gold option
    and make the question's own None candidate the one-hot target.

    Eligible: a None candidate is present, the target is one-hot on a non-None
    option, and at least one other option remains. Soft targets, Score and Noul
    are never changed. The draw depends only on (seed, epoch, record ID), so a
    resumed run sees exactly the same views. Returns (record, relabelled count).
    """
    if prob <= 0:
        return encoded, 0
    rng = _rng(seed, epoch, encoded.record.id)
    questions, changed = [], 0
    for q in encoded.questions:
        question = q.question
        draw = rng.random()  # drawn for every question: other questions keep their stream
        none = none_candidate(question)
        target = question.target
        if (draw >= prob or none is None or target is None or len(target) < 3
                or sorted(target) != [0.0] * (len(target) - 1) + [1.0]):
            questions.append(q)
            continue
        gold = target.index(1.0)
        index = question.candidates.index(none)
        if gold == index:
            questions.append(q)
            continue
        keep = [i for i in range(len(target)) if i != gold]
        question = replace(question, candidates=tuple(question.candidates[i] for i in keep),
                           target=tuple(float(i == index) for i in keep), label=none.id)
        questions.append(replace(q, question=question, candidates=tuple(q.candidates[i] for i in keep)))
        changed += 1
    if not changed:
        return encoded, 0
    return replace(encoded, questions=tuple(questions)), changed


class NoneInserter:
    """Training-only None insertion for records encoded with the as-provided policy.

    Per epoch, each Choice question without a None candidate gets one with
    probability prob (wording drawn from NONE_VARIANTS, skipping IDs already
    used); with conditional probability absent_frac and a one-hot non-None
    target, the gold option is then deleted and the new None becomes the
    target. Evaluation sees the data as provided, so ordinary questions carry
    no extra None option. Draws depend only on (seed, epoch, record ID).
    """
    def __init__(self, encoder, prob, absent_frac):
        if not 0 <= prob < 1 or not 0 <= absent_frac <= 1:
            raise ValueError("invalid None insertion probabilities")
        self.limits, self.prob, self.absent_frac = encoder.limits, prob, absent_frac
        self.variants = [(Candidate(key, key + ": " + text), encoder.candidate_tokens(key + ": " + text))
                         for key, text in NONE_VARIANTS]

    def __call__(self, encoded, *, seed, epoch):
        if self.prob <= 0:
            return encoded, 0, 0
        rng = _rng(seed, f"insert-{epoch}", encoded.record.id)
        questions, inserted, absent = [], 0, 0
        for q in encoded.questions:
            question = q.question
            draw, choice, drop = rng.random(), rng.randrange(len(self.variants)), rng.random()
            if draw >= self.prob or question.type != "choice" or none_candidate(question) is not None:
                questions.append(q)
                continue
            used = {c.id for c in question.candidates}
            options = [v for v in self.variants[choice:] + self.variants[:choice] if v[0].id not in used]
            fits = len(q.candidates) < self.limits.max_candidates
            if not options or not fits:
                questions.append(q)
                continue
            none, tokens = options[0]
            if (len(tokens) > self.limits.max_candidate
                    or len(encoded.state) + len(q.prefix) + len(tokens) > self.limits.max_path):
                questions.append(q)
                continue
            target = question.target
            candidates, rows = question.candidates + (none,), q.candidates + (tokens,)
            new_target = None if target is None else target + (0.0,)
            label = question.label
            hard = (target is not None and len(target) >= 2
                    and sorted(target) == [0.0] * (len(target) - 1) + [1.0])
            if hard and drop < self.absent_frac:
                gold = target.index(1.0)
                keep = [i for i in range(len(candidates)) if i != gold]
                candidates, rows = tuple(candidates[i] for i in keep), tuple(rows[i] for i in keep)
                new_target, label = tuple(float(c == none) for c in candidates), none.id
                absent += 1
            question = replace(question, candidates=candidates, target=new_target, label=label)
            questions.append(replace(q, question=question, candidates=rows))
            inserted += 1
        if not inserted:
            return encoded, 0, 0
        return replace(encoded, questions=tuple(questions)), inserted, absent
