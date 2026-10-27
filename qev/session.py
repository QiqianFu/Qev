"""Incremental state: ask questions while the state grows, reusing its prefix cache."""
from .encoding import ContextOverflow
from .predict import format_predictions
from .schema import Record, render, typed_question


class QevSession:
    """A state S1 + S2 + ... whose state-only cache is extended, never recomputed.

    The state tokens are the concatenation of each chunk's own tokenization, so a
    chunk boundary can differ from tokenizing the joined text in one request; the
    reference and tree executions use the same session tokens. ``cached``
    execution keeps a state-only KV/DeltaNet cache, appends pending chunks on the
    next prediction, and evaluates every question on a fork of it. A session holds
    GPU memory until ``close()`` and is not safe for concurrent use.
    """

    def __init__(self, model, encoder, state='', *, execution='cached', session_id='session'):
        if execution not in {'cached', 'reference', 'tree'}:
            raise ValueError('execution must be cached, reference or tree')
        self.model, self.encoder, self.execution, self.id = model, encoder, execution, str(session_id)
        self._text, self._tokens = '', encoder.state_tokens('')
        self._cache, self._cached = None, 0
        self.extend(state)

    @property
    def state(self):
        return self._text

    @property
    def state_tokens(self):
        return len(self._tokens)

    @property
    def cached_tokens(self):
        return self._cached

    def extend(self, chunk):
        """Append state text; nothing is computed until the next prediction."""
        text = render(chunk)
        tokens = self.encoder.text(text)
        if len(self._tokens) + len(tokens) > self.encoder.limits.max_state:
            raise ContextOverflow(f'{self.id}: state length {len(self._tokens) + len(tokens)} '
                                  f'> {self.encoder.limits.max_state}')
        self._text += text
        self._tokens += tokens
        return self

    def predict(self, questions):
        """Answer a ``questions`` mapping (the request field) over the current state."""
        record = Record(self.id, self.id, 'inference', self._text,
                        tuple(typed_question(qid, q) for qid, q in questions.items()))
        encoded = self.encoder(record, state=self._tokens)
        if self.execution == 'cached':
            self._sync()
            predictions = self.model.predict_state_cache(encoded, self._cache)
        else:
            predictions = self.model.predict(encoded, cached=self.execution == 'tree',
                                             tree=self.execution == 'tree')
        return format_predictions(encoded, predictions)

    def close(self):
        self._cache, self._cached = None, 0

    def _sync(self):
        pending = self._tokens[self._cached:]
        if not pending:
            return
        try:
            self._cache = self.model.extend_state_cache(self._cache, pending, self._cached)
        except BaseException:
            # A failed forward may leave some layers extended: rebuild on next use.
            self.close()
            raise
        self._cached = len(self._tokens)
