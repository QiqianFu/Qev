# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Control for split prefill non-additivity and time the actual branch stages."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import DynamicCache

from .base_evaluate import answer_codes, make_prompt
from .checkpoint import load_model
from .data import file_hash, load_records, write_json


def stats(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


class Timeline:
    """CUDA-stream spans, not active-kernel-only time; Host call wall time (including implicit synchronization) is separate."""
    def __init__(self):
        self.events = []

    @contextmanager
    def span(self, name, metadata=None):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t = time.perf_counter()
        try:
            yield
        finally:
            cpu_ms = 1000*(time.perf_counter()-t)
            end.record()
            self.events.append((name,start,end,cpu_ms,metadata or {}))

    def finish(self):
        return [{'name': n, 'cuda_span_ms': s.elapsed_time(e), 'host_call_ms': c, **m}
                for n,s,e,c,m in self.events]


def cache_bytes(cache):
    seen = set()
    sizes = defaultdict(int)
    for layer in cache.layers:
        for name in ('keys','values','conv_states','recurrent_states'):
            value = getattr(layer,name,None)
            tensors = value.values() if isinstance(value,dict) else [value]
            for t in tensors:
                if isinstance(t,torch.Tensor) and id(t) not in seen:
                    seen.add(id(t))
                    sizes[name] += t.numel()*t.element_size()
    return dict(sizes)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    assert a.repeats >= 1 and torch.cuda.device_count()==1
    a.out.mkdir(parents=True,exist_ok=False)
    model,tok,encoder,meta = load_model(a.checkpoint,'cuda')
    model.prepare_inference(merge_lora=True)
    records,manifest = load_records(a.data,'mmlupro')
    assert manifest['files']['mmlupro']['role'] != 'test'
    encoded = [encoder(r) for r in records]
    ten = sorted([e for e in encoded if len(e.questions[0].candidates)==10],
                 key=lambda e: len(e.state)+len(e.questions[0].prefix))
    sample = [ten[i*(len(ten)-1)//4] for i in range(5)]
    q=records[0].questions[0]
    labels,_,_,suffix=answer_codes(tok,len(q.candidates))
    seed=tok.encode(make_prompt(records[0],q,labels,suffix),add_special_tokens=False)
    ids=tuple((seed*((1024+len(seed)-1)//len(seed)))[:1024])
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'checkpoint':a.checkpoint,'merged_lora':model.lora_merged,'weights_dtype':model.spec.weights_dtype,
        'manifest_sha256':file_hash(Path(a.data)/'manifest.json'),'repeats':a.repeats,
        'scope':'Same merged text backbone, no LM/decision head in sequence controls (identical final hidden-state CPU readback included). Synthetic length sweep is not an accuracy evaluation. Actual request profiling separately includes the joint layer and head.',
        'event_scope':'CUDA-stream wall spans include idle gaps waiting for CPU submission; not active kernel compute time.',
        'sweep':{},'split_controls':{},'requests':[]}

    def measure(fn):
        for _ in range(2):
            fn(None)
        torch.cuda.synchronize()
        times,spans,last = [],[],None
        for _ in range(a.repeats):
            timeline=Timeline()
            torch.cuda.synchronize()
            t=time.perf_counter()
            with timeline.span('total'):
                last=fn(timeline)
            torch.cuda.synchronize()
            times.append(1000*(time.perf_counter()-t))
            spans.append(timeline.finish())
        names={e['name'] for row in spans for e in row}
        aggregated={name:{'cuda_span_ms':stats([sum(e['cuda_span_ms'] for e in row if e['name']==name) for row in spans]),
                          'host_call_ms':stats([sum(e['host_call_ms'] for e in row if e['name']==name) for row in spans])}
                    for name in names}
        return {'wall_ms':stats(times),'stages':aggregated,'raw':spans},last

    def full(n,cached):
        def run(timeline):
            cache=DynamicCache(config=model.backbone.config) if cached else None
            return model._run_rows([ids[:n]],cache=cache).last_hidden_state[0,-1].float().cpu()
        return run

    # Both controls include the same tiny final-state readback (no LM head).
    for n in (32,64,128,256,512,1024):
        report['sweep'][str(n)]={}
        for cached in (False,True):
            result,_=measure(full(n,cached))
            report['sweep'][str(n)]['cache' if cached else 'no_cache']=result
        print(json.dumps({'stage':'length_sweep','tokens':n,
            **{k:v['wall_ms']['median'] for k,v in report['sweep'][str(n)].items()}}),flush=True)
        write_json(a.out/'report.json',report)

    for total,cut in ((256,64),(256,128),(256,192),(1024,512)):
        baseline,reference=measure(full(total,False))

        def split(timeline):
            cache=DynamicCache(config=model.backbone.config)
            if timeline is None:
                model._run_rows([ids[:cut]],cache=cache)
                output=model._run_rows([ids[cut:total]],offset=cut,cache=cache)
            else:
                with timeline.span('prefix',{'tokens':cut}):
                    model._run_rows([ids[:cut]],cache=cache)
                with timeline.span('suffix',{'tokens':total-cut}):
                    output=model._run_rows([ids[cut:total]],offset=cut,cache=cache)
            return output.last_hidden_state[0,-1].float().cpu()

        result,value=measure(split)
        assert torch.isfinite(value).all() and torch.isfinite(reference).all()
        result['paired_full_wall_ms']=baseline['wall_ms']
        result['split_over_full_median_ratio']=result['wall_ms']['median']/baseline['wall_ms']['median']
        result['last_hidden_max_abs_vs_unsplit']=float((value-reference).abs().max())
        report['split_controls'][f'{cut}+{total-cut}']=result
        print(json.dumps({'stage':'split_control','tokens':total,'split':cut,'wall_ms':result['wall_ms'],
                          'stages':result['stages']}),flush=True)
        write_json(a.out/'report.json',report)

    for e in sample:
        untraced,_=measure(lambda _:model.predict(e))
        original={name:getattr(model,name) for name in ('_run_rows','fork_cache','_cached_joint_features','_probabilities')}
        active=None

        def traced_rows(rows,**kwargs):
            if active is None:
                return original['_run_rows'](rows,**kwargs)
            name='candidate_backbone' if kwargs.get('offset',0)>0 else 'prefix_backbone'
            with active.span(name,{'batch':len(rows),'width':max(map(len,rows)),
                                  'tokens':sum(map(len,rows))}):
                return original['_run_rows'](rows,**kwargs)

        def traced_fork(cache,copies=1):
            if active is None:
                return original['fork_cache'](cache,copies)
            with active.span('cache_fork',{'copies':copies,'parent_bytes':cache_bytes(cache)}):
                return original['fork_cache'](cache,copies)

        def traced_joint(*args,**kwargs):
            if active is None:
                return original['_cached_joint_features'](*args,**kwargs)
            with active.span('joint_final_layer'):
                return original['_cached_joint_features'](*args,**kwargs)

        def traced_head(*args,**kwargs):
            if active is None:
                return original['_probabilities'](*args,**kwargs)
            with active.span('decision_head'):
                return original['_probabilities'](*args,**kwargs)

        model._run_rows,model.fork_cache=traced_rows,traced_fork
        model._cached_joint_features,model._probabilities=traced_joint,traced_head
        try:
            def predict(timeline):
                nonlocal active
                active=timeline
                try:
                    return model.predict(e)
                finally:
                    active=None
            result,_=measure(predict)
        finally:
            for name,method in original.items():
                setattr(model,name,method)
        q=e.questions[0]
        result.update({'record_id':e.record.id,'prefix_tokens':len(e.state)+len(q.prefix),
            'candidate_tokens':list(map(len,q.candidates)),'untraced_wall_ms':untraced['wall_ms'],
            'instrumentation_median_ratio':result['wall_ms']['median']/untraced['wall_ms']['median']})
        report['requests'].append(result)
        print(json.dumps({'stage':'request_profile',**{k:v for k,v in result.items() if k!='raw'}}),flush=True)
        write_json(a.out/'report.json',report)
    report['completed']=True
    write_json(a.out/'report.json',report)


if __name__=='__main__':
    main()
