# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict, replace
import json

import pytest
import torch
from safetensors.torch import load_file

from qev.checkpoint import load_model, save_model
from qev.data import write_json
from qev.distill import train
from qev.distillation import TeacherCache, training_views, view_protocol
from qev.encoding import Encoder, Limits
from qev.model import QevModel, ModelSpec, load_backbone
from qev.refinement import distribution_loss, microbatches, refinement_plan
from qev.response_data import load_response_data
from qev.teacher import cache_logits, cache_responses
from conftest import tiny_backbone


def test_schedule_counts_endpoints_and_never_splits_a_pair():
    plan=refinement_plan(7,['source']*40,12,32,.75,17)
    assert plan==refinement_plan(7,['source']*40,12,32,.75,17)
    assert plan!=refinement_plan(7,['source']*40,12,32,.75,18)
    for batch in plan:
        assert sum(k=='pair' for k,_ in batch)==4
        assert sum(k=='replay' for k,_ in batch)==24
        batches=list(microbatches(batch,16))
        assert [item for micro in batches for item in micro]==batch
        assert all(sum(2 if k=='pair' else 1 for k,_ in micro)<=16 for micro in batches)


def test_soft_cross_entropy_matches_explicit_teacher_temperature():
    z=torch.tensor([.5,-.2,1.1],requires_grad=True)
    teacher=torch.tensor([1.,-1.,.3])
    expected=-(teacher.div(1.563437713227029).softmax(-1)*z.log_softmax(-1)).sum()
    actual=distribution_loss(z,teacher,teacher_temperature=1.563437713227029)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.backward();assert z.grad.abs().sum()>0


def assets(tmp_path, tokenizer, record):
    torch.manual_seed(12)
    base=tmp_path/'base';tiny_backbone(len(tokenizer),hybrid=True).save_pretrained(base);tokenizer.save_pretrained(base)
    spec=ModelSpec(str(base),head_dim=32,head_heads=4,head_layers=1,lora_rank=2,
                   weights_dtype='fp32',attention='eager')
    teacher=QevModel(load_backbone(spec,'cpu'),spec,tokenizer.pad_token_id).eval()
    limits=Limits(256,128,128,512,32);encoder=Encoder(tokenizer,limits,choice_none_policy='as-provided')
    student=tmp_path/'student';save_model(student,teacher,tokenizer,limits)
    data=tmp_path/'data';data.mkdir()
    rows=[replace(record,id=f'r{i}',group_id=f'g{i}',source='custom',
                  state=f'The parcel weighs {i+3} kg. The limit is at most 9 kg. Packaging is blue.',
                  questions=(record.questions[0],)) for i in range(10)]
    (data/'train.jsonl').write_text(''.join(json.dumps(r.to_dict())+'\n' for r in rows))
    write_json(data/'manifest.json',{'files':{'train':{'file':'train.jsonl','role':'train','records':len(rows)}}})
    config={'model':asdict(spec),'limits':asdict(limits),'training':{
        'epochs':2,'seed':17,'none_insert_prob':.99,'none_insert_absent_frac':.5}}
    responses=tmp_path/'responses';responses.mkdir()
    cache_responses(data,responses,config,encoder,teacher,encoder,confidence=0,exclude_sources=[])
    return data,config,encoder,teacher,student,responses


def test_teacher_preparation_covers_augmented_views_and_keeps_labels_out(tmp_path,tokenizer,record):
    data,config,encoder,teacher,student,responses=assets(tmp_path,tokenizer,record)
    out=tmp_path/'logits';out.mkdir()
    cache_logits(data,out,config,encoder,teacher,encoder)
    from qev.data import load_records
    records=load_records(data,'train',training=True)[0]
    encoded=[encoder(r) for r in records]
    cache=TeacherCache(out,protocol=view_protocol(config))
    assert cache.verify_coverage(training_views(encoded,len(encoded),encoder,config))['complete_coverage']
    rows,manifest=load_response_data(responses)
    assert manifest['response_scale']>=1e-6
    assert rows['pairs'] and rows['replay']
    for row in rows['pairs']:
        assert row['original']['questions'][0]['target'] is None
        assert row['edited']['questions'][0]['label'] is None
    # Re-signing or editing a manifest must not admit original-label supervision.
    q=rows['replay'][0]['record']['questions'][0]
    q['label']=q['candidates'][0]['id']
    (responses/'replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows['replay']))
    with pytest.raises(ValueError,match='original labels'):
        load_response_data(responses)


def test_response_training_cpu_resume_and_model_reload(tmp_path,tokenizer,record):
    _,_,_,_,student,data=assets(tmp_path,tokenizer,record)
    cfg={'steps':3,'batch_endpoints':4,'microbatch_endpoints':2,'replay_fraction':.5,'seed':17,
         'lr':.001,'head_lr':.002,'gate_lr':.001,'weight_decay':.01,'warmup_steps':1,
         'minimum_lr_fraction':.1,'gradient_clip':1.,'save_steps':[1,3],
         'teacher_temperature':1.563437713227029,'student_temperature':1.,
         'response_weight':.1,'gradient_checkpointing':False}
    full=train(student,data,cfg,tmp_path/'full',device='cpu')
    first=train(student,data,cfg,tmp_path/'resumed',device='cpu',stop_after=1)
    resumed=train(student,data,cfg,tmp_path/'resumed',device='cpu',resume=first)
    for name in ('adapter/adapter_model.safetensors','head.safetensors'):
        a,b=load_file(str(full/name)),load_file(str(resumed/name))
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    model,_,encoder,_=load_model(resumed,'cpu')
    model.eval()
    assert all(torch.isfinite(p).all() for p in model.predict(encoder(record),cached=False))
    changed=dict(cfg,teacher_temperature=2.)
    with pytest.raises(ValueError,match='configuration or training data'):
        train(student,data,changed,tmp_path/'changed',device='cpu',resume=first)


@pytest.mark.parametrize('unlabelled,temperature', [(False, 1.), (True, 1.563437713227029)])
def test_probability_distillation_uses_standard_trainer_and_detects_target_changes(tmp_path,tokenizer,record,unlabelled,temperature):
    from test_training import training_assets, run_training
    data,config_path=training_assets(tmp_path,tokenizer,record)
    config=json.loads(config_path.read_text())
    cache=tmp_path/'teacher';cache.mkdir()
    config['training']['distillation']={'cache':str(cache),'weight':1.0,'temperature':temperature}
    config['training']['reset_training_rng']=True
    if unlabelled:
        rows=[json.loads(line) for line in (data/'train.jsonl').read_text().splitlines()]
        for row in rows:
            for question in row['questions']:
                question.pop('target', None)
                question.pop('label', None)
        (data/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        manifest=json.loads((data/'manifest.json').read_text())
        manifest['files']['train'].pop('sha256')
        (data/'manifest.json').write_text(json.dumps(manifest))
    config_path.write_text(json.dumps(config))
    spec=ModelSpec(**config['model'])
    teacher=QevModel(load_backbone(spec,'cpu'),spec,tokenizer.pad_token_id).eval()
    encoder=Encoder(tokenizer,Limits(**config['limits']),choice_none_policy=spec.choice_none_policy)
    cache_logits(data,cache,config,encoder,teacher,encoder)
    out=tmp_path/'training'
    run_training(data,config_path,out,'--max-steps','1')
    assert json.loads((out/'distillation.json').read_text())['temperature']==temperature
    run_training(data,config_path,out,'--resume',str(out/'step-000001'),'--max-steps','2')
    rows=[json.loads(line) for line in (cache/'logits.jsonl').read_text().splitlines()]
    rows[0]['logits'][0]+=.5
    (cache/'logits.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(AssertionError,match='resume data or admitted sample set changed'):
        run_training(data,config_path,out,'--resume',str(out/'step-000002'))
