# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Two-level candidate tree; logical positions never depend on sibling order."""
from dataclasses import dataclass
import re
from .schema import Record, Question
from .choice_policy import ALWAYS, POLICIES, ensure_choice_none, none_candidate

# Reserved delimiter convention from Kev (Apache-2.0); user text cannot forge them.
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
LAYOUT = "state-question-candidate.v1"
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass(frozen=True)
class Limits:
    max_state: int = 1024
    max_question: int = 512
    max_candidate: int = 256
    max_path: int = 2048
    max_candidates: int = 128


class ContextOverflow(ValueError):
    pass


@dataclass(frozen=True)
class EncodedQuestion:
    question: Question
    prefix: tuple[int, ...]  # Q only, shared by its candidates
    candidates: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EncodedRecord:
    record: Record
    state: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]

    @property
    def forward_tokens(self):
        return sum(len(self.state) + len(q.prefix) + len(c) for q in self.questions for c in q.candidates)


class Encoder:
    def __init__(self, tokenizer, limits=Limits(), *, choice_none_policy="as-provided"):
        if choice_none_policy not in POLICIES:
            raise ValueError("unknown Choice None policy")
        self.choice_none_policy = choice_none_policy
        self.tokenizer = tokenizer
        self.limits = limits
        self.special = tuple(tokenizer.convert_tokens_to_ids(t) for t in SPECIAL)
        if len(set(self.special)) != len(SPECIAL) or any(i is None or i == tokenizer.unk_token_id for i in self.special):
            raise ValueError("tokenizer must contain the five reserved Qwen delimiter tokens")
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

    def text(self, text):
        return tuple(self.tokenizer.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False))

    def candidate_tokens(self, text):
        _, _, op, end, _ = self.special
        return (op,) + self.text(text) + (end,)

    def __call__(self, record):
        if self.choice_none_policy in ALWAYS:
            for q in record.questions:
                if q.type == "choice" and none_candidate(q) is None and len(q.candidates) >= min(255, self.limits.max_candidates):
                    raise ContextOverflow(f"{record.id}/{q.id}: no room for required None candidate")
            record = ensure_choice_none(record, varied=self.choice_none_policy == "always-varied")
        st, qu, op, end, read = self.special
        state = (st,) + self.text(record.state)
        if len(state) > self.limits.max_state:
            raise ContextOverflow(f"{record.id}: state length {len(state)} > {self.limits.max_state}")
        questions = []
        for q in record.questions:
            prefix = (qu,) + self.text(q.type + " question: " + q.instructions) + (read,)
            candidates = tuple(self.candidate_tokens(c.text) for c in q.candidates)
            if len(prefix) > self.limits.max_question or len(candidates) > self.limits.max_candidates:
                raise ContextOverflow(f"{record.id}/{q.id}: question or candidate-count limit")
            if any(len(c) > self.limits.max_candidate or len(state) + len(prefix) + len(c) > self.limits.max_path
                   for c in candidates):
                raise ContextOverflow(f"{record.id}/{q.id}: candidate or root-to-leaf limit")
            questions.append(EncodedQuestion(q, prefix, candidates))
        return EncodedRecord(record, state, tuple(questions))


def leaf_rows(encoded):
    """Independent causal rows are the gradient-correct reference for hybrid LMs."""
    rows, pointers = [], []
    for qi, q in enumerate(encoded.questions):
        prefix = encoded.state + q.prefix
        for ci, candidate in enumerate(q.candidates):
            ids = prefix + candidate
            rows.append(ids)
            pointers.append((qi, ci, len(prefix) - 1, len(ids) - 1))
    return rows, pointers


def packed_tree(encoded):
    """Attention-only oracle, not a valid DeltaNet execution path.

    Returns physical IDs, path-depth positions, and a boolean allow matrix.
    Kept independent of torch so the information boundary is directly inspectable.
    """
    ids, pos, owners, local = [], [], [], []
    ancestors = {0: set()}

    def add(tokens, node, start):
        ids.extend(tokens)
        pos.extend(range(start, start + len(tokens)))
        owners.extend([node] * len(tokens))
        local.extend(range(len(tokens)))

    add(encoded.state, 0, 0)
    node = 1
    for q in encoded.questions:
        parent = node
        ancestors[parent] = {0}
        add(q.prefix, parent, len(encoded.state))
        node += 1
        for c in q.candidates:
            ancestors[node] = {0, parent}
            add(c, node, len(encoded.state) + len(q.prefix))
            node += 1
    allow = [[owners[v] in ancestors[owners[u]] or (owners[v] == owners[u] and local[v] <= local[u])
              for v in range(len(ids))] for u in range(len(ids))]
    return ids, pos, allow
