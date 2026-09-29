# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Target-independent, checkpointed handling of the none-of-the-above option."""
from dataclasses import replace
import hashlib
import re

from .schema import Candidate

POLICIES = ('as-provided', 'always', 'always-varied')
ALWAYS = ('always', 'always-varied')
NONE_ID = 'none_of_these'
NONE_TEXT = 'None of these options describes the answer'
# 'always-varied' draws the appended None's wording from this pool, keyed only by
# record/question IDs (never labels), so no single phrasing is always negative.
NONE_VARIANTS = (
    (NONE_ID, NONE_TEXT),
    (NONE_ID, 'None of these options is correct.'),
    ('none_of_the_above', 'None of the listed options matches the answer.'),
    ('not_listed', 'The correct answer is not among the listed options.'),
    ('no_match', 'None of the available options applies.'),
    ('other', 'An answer not covered by any of the other options.'),
)
_IDS = {NONE_ID, 'none_of_the_above', 'none_of_above', 'e_none_of_above', 'not_listed', 'no_match'}
_TEXTS = {
    'none of these', 'none of the above', 'none of these options is correct',
    'none of these options describes the answer', 'none of the listed options matches the answer',
    'none of the listed options apply', 'none of the available options applies',
    'the correct answer is not among the listed options',
    'an answer not covered by any of the other options',
    'an answer not covered by the other options', 'no option matches',
}


def none_candidate(question):
    if question.type != 'choice':
        return None
    matches = []
    for candidate in question.candidates:
        key = re.sub(r'[^a-z0-9]+', '_', candidate.id.casefold()).strip('_')
        text = candidate.text
        if text.startswith(candidate.id + ':'):
            text = text[len(candidate.id)+1:]
        normalized = ' '.join(text.casefold().strip(' .!').split())
        if key in _IDS or normalized in _TEXTS or (key == 'none' and normalized == 'none'):
            matches.append(candidate)
    if len(matches) > 1:
        raise ValueError(f'{question.id}: multiple none-of-the-above candidates are ambiguous')
    return matches[0] if matches else None


def _variant(record_id, question_id, taken):
    digest = hashlib.sha256(f"{record_id}\0{question_id}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % len(NONE_VARIANTS)
    for offset in range(len(NONE_VARIANTS)):
        key, text = NONE_VARIANTS[(start + offset) % len(NONE_VARIANTS)]
        if key not in taken:
            return key, text
    raise ValueError(f'{question_id}: every None candidate ID is already used')


def ensure_choice_none(record, varied=False):
    """Keep existing explicit None IDs/rubrics; append one only when missing.

    Labels do not influence candidate detection or insertion. Known hard/soft
    targets retain their original masses and assign zero to the added option.
    Evidence-free uniform targets remain uniform over their original candidates;
    uncertainty is not re-labelled as none-of-the-above.
    """
    questions = []
    for question in record.questions:
        if question.type != 'choice' or none_candidate(question) is not None:
            questions.append(question)
            continue
        key, text = (_variant(record.id, question.id, {c.id for c in question.candidates})
                     if varied else (NONE_ID, NONE_TEXT))
        questions.append(replace(question,
            candidates=question.candidates + (Candidate(key, key + ': ' + text),),
            target=question.target + (0.0,) if question.target is not None else None))
    return replace(record, questions=tuple(questions))
