"""Small Python API backed by the same checkpoint and prediction code as the CLI."""
from dataclasses import replace

from .checkpoint import load_model
from .encoding import Encoder
from .predict import predict_request


class Qev:
    def __init__(self, model, encoder, *, execution='cached'):
        if execution not in {'cached', 'reference', 'tree'}:
            raise ValueError('execution must be cached, reference or tree')
        self.model, self.encoder, self.execution = model, encoder, execution
        self.model.prepare_inference()

    @classmethod
    def from_pretrained(cls, checkpoint, *, device='cuda', revision=None, base=None,
                        weights_dtype='checkpoint', execution='cached', max_state=None, max_path=None):
        if weights_dtype not in {'checkpoint', 'fp32'}:
            raise ValueError('weights_dtype must be checkpoint or fp32')
        model, tokenizer, encoder, _ = load_model(checkpoint, device, revision=revision, base=base,
                                                weights_dtype='fp32' if weights_dtype == 'fp32' else None)
        if max_state is not None or max_path is not None:
            limits = replace(encoder.limits, max_state=max_state or encoder.limits.max_state,
                             max_path=max_path or encoder.limits.max_path)
            encoder = Encoder(tokenizer, limits, choice_none_policy=encoder.choice_none_policy)
        return cls(model, encoder, execution=execution)

    def predict(self, request):
        return predict_request(self.model, self.encoder, request,
                               cached=self.execution == 'cached', tree=self.execution == 'tree')
