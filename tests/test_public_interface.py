import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qev.prepare import build_dataset
from qev.data import load_records, file_hash
from qev.schema import Record
from qev.artifacts import resolve_checkpoint
from qev.export import export_checkpoint
from qev.model import ModelSpec, QevModel, load_backbone
from qev.checkpoint import save_model, load_model
from qev.encoding import Limits
from qev import Qev
from conftest import tiny_backbone


def request(rid, state, group=None):
    return {'id':rid, 'group_id':group or rid, 'state':state, 'questions':{
        'route':{'type':'choice', 'instructions':'Select a team.',
                 'criteria':{'billing':'Payments','delivery':'Shipping'}, 'label':'billing'}}}


def test_prepare_group_split_and_legacy_loading(tmp_path, record):
    path=tmp_path/'input.jsonl'
    rows=[request(f'r{i}',f'A payment question {i}.',f'g{i//2}') for i in range(8)]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    out=build_dataset(path,tmp_path/'data',validation_fraction=.25)
    train,_=load_records(out,'train',training=True);dev,_=load_records(out,'dev')
    assert len(train)==6 and len(dev)==2
    assert not {r.group_id for r in train}&{r.group_id for r in dev}
    with pytest.raises(ValueError,match='cannot be used'):load_records(out,'dev',training=True)
    old=record.to_dict();old['schema']='branchkev.record.v1'
    assert Record.from_dict(old)==record


def test_prepare_rejects_input_overlap_and_preserves_destination(tmp_path):
    a,b=tmp_path/'a.jsonl',tmp_path/'b.jsonl'
    a.write_text(json.dumps(request('a','Same input.'))+'\n')
    b.write_text(json.dumps(request('b','Same input.'))+'\n')
    with pytest.raises(ValueError,match='overlap in inputs'):build_dataset(a,tmp_path/'data',validation=b)
    assert not (tmp_path/'data').exists()
    b.write_text(json.dumps(request('b','Different input.'))+'\n')
    build_dataset(a,tmp_path/'data',validation=b)
    with pytest.raises(FileExistsError):build_dataset(a,tmp_path/'data',validation=b)


def test_portable_export_legacy_load_and_python_api(tmp_path, tokenizer, record):
    base=tmp_path/'base';tiny_backbone(len(tokenizer)).save_pretrained(base);tokenizer.save_pretrained(base)
    spec=ModelSpec(str(base),head_dim=32,head_heads=4,head_layers=0,lora_rank=2,weights_dtype='fp32',attention='eager')
    model=QevModel(load_backbone(spec,'cpu'),spec,tokenizer.pad_token_id).eval()
    original=tmp_path/'original';save_model(original,model,tokenizer,Limits(128,128,128,384,32))
    (original/'training.pt').write_bytes(b'optimizer must not be distributed')
    (original/'LICENSE').write_text('Model license supplied by its author')
    (original/'licenses').mkdir()
    (original/'licenses/upstream.txt').write_text('Upstream attribution')
    metadata=json.loads((original/'model.json').read_text());metadata['format']='branchkev.checkpoint.v1'
    (original/'model.json').write_text(json.dumps(metadata))
    exported=export_checkpoint(original,tmp_path/'export',base='example/base',revision='release-1')
    assert not (exported/'training.pt').exists()
    assert not (exported/'SHA256SUMS.json').exists()
    assert (exported/'LICENSE').read_bytes()==(original/'LICENSE').read_bytes()
    assert (exported/'licenses/upstream.txt').read_bytes()==(original/'licenses/upstream.txt').read_bytes()
    assert file_hash(original/'head.safetensors')==file_hash(exported/'head.safetensors')
    assert json.loads((exported/'model.json').read_text())['spec']['base']=='example/base'
    loaded,_,enc,_=load_model(exported,'cpu',base=str(base))
    model.eval();loaded.eval();encoded=enc(record)
    for a,b in zip(model.predict(encoded,cached=False),loaded.predict(encoded,cached=False)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    public=Qev.from_pretrained(exported,device='cpu',base=str(base),execution='reference')
    r=request('demo','A payment question.')
    r['questions']['ok']={'type':'noul','instructions':'Is it a payment question?'}
    r['questions']['priority']={'type':'score','instructions':'Rate priority.','criteria':['low','high']}
    out=public.predict(r)
    assert set(out)=={'route','ok','priority'}
    assert 0<=out['ok']['noul']<=1 and 0<=out['priority']['score']<=1
    assert out['route']['choice'] in {'billing','delivery'}


def test_hub_resolution_defaults_to_main_and_accepts_versions(tmp_path, monkeypatch):
    import huggingface_hub
    cache=tmp_path/'snapshot';cache.mkdir();(cache/'model.json').write_text('{}')
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        return str(cache)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    assert resolve_checkpoint('owner/model')==cache
    assert calls[-1]['revision'] is None
    assert resolve_checkpoint('owner/model@v1')==cache
    assert calls[-1]['revision']=='v1'
    assert resolve_checkpoint('owner/model',revision='v2')==cache
    assert calls[-1]['revision']=='v2'
    with pytest.raises(ValueError,match='conflicting'):resolve_checkpoint('owner/model@v1',revision='v2')
    with pytest.raises(ValueError,match='empty'):resolve_checkpoint('owner/model@')
    with pytest.raises(FileNotFoundError):resolve_checkpoint(tmp_path/'missing')


def test_initialize_on_new_data_resets_schedule(tmp_path, tokenizer, record):
    from test_training import training_assets, run_training
    data,config=training_assets(tmp_path,tokenizer,record)
    run_training(data,config,tmp_path/'first','--max-steps','1')
    rows=[replace(record,id=f'new{i}',group_id=f'new-group{i}',state=record.state+' Updated.') for i in range(4)]
    path=data/'train.jsonl';path.write_text(''.join(json.dumps(r.to_dict())+'\n' for r in rows))
    manifest=json.loads((data/'manifest.json').read_text());manifest['files']['train']['sha256']=file_hash(path)
    (data/'manifest.json').write_text(json.dumps(manifest))
    run_training(data,config,tmp_path/'new-domain','--init-checkpoint',str(tmp_path/'first/step-000001'),'--max-steps','1')
    state=torch.load(tmp_path/'new-domain/step-000001/training.pt',weights_only=False)
    assert state['global_step']==1
    init=json.loads((tmp_path/'new-domain/initialization.json').read_text())
    assert init['optimizer']=='fresh' and init['global_step']==0


def test_resume_detects_changed_data_without_manifest_checksums(tmp_path, tokenizer, record):
    from test_training import training_assets, run_training
    data,config=training_assets(tmp_path,tokenizer,record)
    manifest=json.loads((data/'manifest.json').read_text())
    manifest['files']['train'].pop('sha256')
    (data/'manifest.json').write_text(json.dumps(manifest))
    run_training(data,config,tmp_path/'run','--max-steps','1')
    path=data/'train.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['state']+=' Updated.'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(AssertionError, match="resume data or admitted sample set changed"):
        run_training(data,config,tmp_path/'run','--resume',str(tmp_path/'run/step-000001'))
