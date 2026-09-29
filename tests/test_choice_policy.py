# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import asdict, replace
import json

import pytest
import torch

from qev.choice_policy import NONE_ID, ensure_choice_none, none_candidate
from qev.encoding import ContextOverflow, Encoder, Limits, leaf_rows
from qev.schema import Candidate


def test_none_insertion_is_idempotent_and_preserves_targets(record, encoder):
    updated = ensure_choice_none(record)
    q = updated.questions[0]
    assert len(q.candidates) == len(record.questions[0].candidates) + 1
    assert q.label == 'blue' and q.target == (1., 0., 0., 0.)
    assert none_candidate(q).id == NONE_ID
    assert updated.questions[1] == record.questions[1]  # Noul keeps its binary contract.
    assert ensure_choice_none(updated) == updated
    soft = replace(record, questions=(replace(record.questions[0], target=(.2,.3,.5), label=None),))
    assert ensure_choice_none(soft).questions[0].target == (.2,.3,.5,0.)
    plain = encoder(record)
    expanded = Encoder(encoder.tokenizer, encoder.limits, choice_none_policy='always')(record)
    a, b = plain.questions[0], expanded.questions[0]
    assert plain.state == expanded.state and a.prefix == b.prefix and a.candidates == b.candidates[:-1]
    relabelled = replace(record, questions=(replace(record.questions[0], label='red', target=(0.,1.,0.)),record.questions[1]))
    changed = Encoder(encoder.tokenizer, encoder.limits, choice_none_policy='always')(relabelled)
    assert leaf_rows(expanded) == leaf_rows(changed)


def test_existing_none_aliases_and_information_unknown_are_distinct(record):
    q = record.questions[0]
    for key, text in [('E_none_of_above','Domain-specific fallback rubric'),
                      ('other','other: An answer not covered by any of the other options.')]:
        r = replace(record, questions=(replace(q,candidates=q.candidates+(Candidate(key,text),),target=q.target+(0.,)),))
        assert ensure_choice_none(r) == r
        assert none_candidate(r.questions[0]).id == key
    unknown = replace(q,candidates=q.candidates+(Candidate('unknown','Insufficient evidence to decide'),),target=q.target+(0.,))
    normalized = ensure_choice_none(replace(record, questions=(unknown,)))
    assert {c.id for c in normalized.questions[0].candidates} >= {'unknown',NONE_ID}
    duplicate = replace(q,candidates=q.candidates+(Candidate(NONE_ID,'None of these'),Candidate('none_of_above','None of the above')),
                        target=q.target+(0.,0.))
    with pytest.raises(ValueError,match='multiple'):
        ensure_choice_none(replace(record,questions=(duplicate,)))


def test_candidate_limit_accounts_for_required_none(record, tokenizer):
    enc = Encoder(tokenizer,Limits(128,128,128,384,3),choice_none_policy='always')
    with pytest.raises(ContextOverflow,match='required None'):
        enc(record)


def test_inference_returns_none_instead_of_forcing_an_original_label(tokenizer):
    from qev.predict import predict_request
    request = {'state':'An unsupported case.','questions':{'route':{
        'type':'choice','instructions':'Choose a route.','criteria':{'a':'A','b':'B'}}}}
    encoder = Encoder(tokenizer,Limits(128,128,128,384,32),choice_none_policy='always')
    class Model:
        def predict(self, encoded, *, cached, tree=False):
            q = encoded.questions[0].question
            return [torch.tensor([float(c.id==NONE_ID) for c in q.candidates])]
    result = predict_request(Model(),encoder,request)['route']
    assert result['prediction']==NONE_ID and result['predicted_none']
    assert set(result['probabilities'])=={'a','b',NONE_ID}


def test_checkpoint_roundtrip_and_actual_model_outputs(tmp_path, tokenizer, record):
    from conftest import tiny_backbone
    from qev.model import QevModel, ModelSpec, load_backbone
    from qev.checkpoint import save_model, load_model
    base=tmp_path/'base';tiny_backbone(len(tokenizer)).save_pretrained(base);tokenizer.save_pretrained(base)
    spec=ModelSpec(str(base),head_dim=32,head_heads=4,head_layers=0,lora_rank=2,
                   weights_dtype='fp32',choice_none_policy='always',attention='eager')
    model=QevModel(load_backbone(spec,'cpu'),spec,tokenizer.pad_token_id).eval()
    limits=Limits(128,128,128,384,32)
    save_model(tmp_path/'checkpoint',model,tokenizer,limits)
    loaded,_,encoder,meta=load_model(tmp_path/'checkpoint')
    assert loaded.spec.choice_none_policy==encoder.choice_none_policy==meta['spec']['choice_none_policy']=='always'
    loaded.eval();encoded=encoder(record)
    prediction=loaded.predict(encoded,cached=False)
    assert prediction[0].shape==(4,) and prediction[1].shape==(2,)
    assert torch.isfinite(sum(prediction[0])).all()
    with pytest.raises(ValueError,match='saved Encoder'):
        loaded.predict(Encoder(tokenizer,limits)(record),cached=False)
