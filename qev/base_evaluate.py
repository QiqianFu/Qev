# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Frozen Qwen native-LM-head baseline, with exact multi-token answer-code scores."""
import argparse
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import time

import torch

from .data import load_records, write_json
from .evaluate import checked_probabilities, summarize


def answer_codes(tokenizer, count):
    if count <= 26:
        labels = [chr(65+i) for i in range(count)]
        texts = [" " + label for label in labels]
        suffix = "\nAnswer:"
    else:
        width = len(str(count-1))
        labels = [str(i).zfill(width) for i in range(count)]
        texts = labels
        suffix = "\nAnswer: "
    tokens = [tuple(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    if not all(tokens) or len(set(tokens)) != count or len({len(t) for t in tokens}) != 1:
        raise ValueError("answer codes must be unique, nonempty and equally many tokens")
    if count <= 26 and any(len(t) != 1 for t in tokens):
        raise ValueError("this tokenizer does not have single-token letter answers")
    return labels, texts, tokens, suffix


def make_prompt(record, question, labels, suffix):
    def escape(s):
        return re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", s)
    options = "\n".join(f"{label}. {escape(c.text)}" for label,c in zip(labels,question.candidates,strict=True))
    return ("Read the evidence and answer the question. Choose exactly one listed option. "
            "Respond with only its option label.\n\nEvidence:\n" + escape(record.state)
            + "\n\nQuestion:\n" + escape(question.instructions) + "\n\nOptions:\n" + options + suffix)


@torch.no_grad()
def score_codes(model, prompt_ids, sequences, device):
    """Log P(complete code | prompt), preserving each branch's independent cache."""
    multi = any(len(s)>1 for s in sequences)
    output = model(input_ids=torch.tensor([prompt_ids],device=device), use_cache=multi, logits_to_keep=1)
    scores = [None]*len(sequences)
    def visit(items, logits, cache, prefix_score):
        lp = logits.float().log_softmax(-1)
        groups = defaultdict(list)
        for index,seq in items:
            if len(seq)==1:
                scores[index]=prefix_score+lp[seq[0]]
            else:
                groups[seq[0]].append((index,seq[1:]))
        for token,children in groups.items():
            child = model(input_ids=torch.tensor([[token]],device=device),
                          past_key_values=copy.deepcopy(cache), use_cache=True, logits_to_keep=1)
            visit(children,child.logits[0,-1],child.past_key_values,prefix_score+lp[token])
    visit(list(enumerate(sequences)),output.logits[0,-1],output.past_key_values,0.0)
    return torch.stack(scores)


@torch.no_grad()
def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--model",default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--revision",help="optional model version")
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--max-prompt",type=int,default=8192)
    p.add_argument("--smoke",action="store_true",help="Only largest-prompt question in each split, in a separate diagnostic run")
    p.add_argument("--splits",nargs="+",default=["decision_dev","transfer_dev"],help="non-test splits to evaluate")
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    pools={}
    for s in a.splits:
        records,manifest=load_records(a.data,s)
        if manifest['files'][s]['role']=='test':
            p.error('test splits are not evaluated here')
        pools[s]=records
    prepared={}
    for split,records in pools.items():
        items=[]
        for r in records:
            for q in r.questions:
                labels,texts,codes,suffix=answer_codes(tok,len(q.candidates))
                prompt=make_prompt(r,q,labels,suffix)
                ids=tok.encode(prompt,add_special_tokens=False)
                # Check the real prompt/answer boundary, rather than assuming token concatenation.
                for index in {0,len(codes)-1}:
                    assert tok.encode(prompt+texts[index],add_special_tokens=False)==ids+list(codes[index])
                items.append((r,q,ids,codes,labels,prompt))
        prepared[split]=[max(items,key=lambda x:len(x[2]))] if a.smoke else items
    write_json(out/'protocol.json',{'model':a.model,'revision':a.revision,
        'native_lm_head':True,'adapter':False,'prompt':'plain zero-shot, no chat template or reasoning generation',
        'readout':'single-token letter for K<=26; fixed-width numeric code with full joint log probability for K>26',
        'dtype':'bf16','temperature':1.0,'max_prompt':a.max_prompt,'smoke':a.smoke,
        'largest_prompts':{s:max(len(x[2]) for x in v) for s,v in prepared.items()}})
    model=AutoModelForImageTextToText.from_pretrained(a.model,revision=a.revision,dtype=torch.bfloat16,
            attn_implementation='sdpa',trust_remote_code=False)
    # The benchmark is text-only; preserve the original text tower AND lm_head.
    model.model.visual=None
    model.requires_grad_(False).eval().to('cuda')
    assert model.get_output_embeddings() is not None
    for split,items in prepared.items():
        dest=out/split;dest.mkdir()
        predictions=[];rejected=[];rejected_source=defaultdict(int);total_seconds=0
        with (dest/'predictions.jsonl').open('w') as stream:
            for index,(r,q,ids,codes,labels,prompt) in enumerate(items):
                if len(ids)+max(map(len,codes))-1>a.max_prompt:
                    rejected.append({'id':r.id,'question_id':q.id,'source':r.source,'questions':1,'reason':'native prompt token limit'})
                    rejected_source[r.source]+=1;continue
                torch.cuda.synchronize();start=time.perf_counter()
                scores=score_codes(model,ids,codes,'cuda')
                assert torch.isfinite(scores).all()
                probabilities=checked_probabilities(scores.softmax(-1),len(codes))
                torch.cuda.synchronize();seconds=time.perf_counter()-start;total_seconds+=seconds
                candidate_ids=[c.id for c in q.candidates]
                pred=candidate_ids[int(probabilities.argmax())]
                target=torch.tensor(q.target) if q.target is not None else None
                row={'record_id':r.id,'group_id':r.group_id,'question_id':q.id,'source':r.source,'type':q.type,
                     'label':q.label,'prediction':pred,'correct':pred==q.label if q.label is not None else None,
                     'probabilities':dict(zip(candidate_ids,probabilities.tolist())), 'confidence':float(probabilities.max()),
                     'brier':float((probabilities-target).square().sum()) if target is not None else None,
                     'nll':float(-(target*probabilities.clamp_min(1e-12).log()).sum()) if target is not None else None,
                     'answer_codes':dict(zip(candidate_ids,labels)),'answer_code_tokens':list(map(list,codes)),
                     'native_code_log_probabilities':scores.float().cpu().tolist(),
                     'native_total_valid_code_mass':float(scores.logsumexp(0).exp()),
                     'prompt_tokens':len(ids),'question_seconds':seconds}
                predictions.append(row);stream.write(json.dumps(row)+'\n')
                if (index+1)%50==0 or a.smoke:
                    stream.flush();print(json.dumps({'split':split,'questions':index+1,'total':len(items),'seconds':total_seconds,
                        'peak_gpu_bytes':torch.cuda.max_memory_allocated()}),flush=True)
        report=summarize(predictions,len(rejected),rejected_source)
        report.update({'split':split,'rejected':rejected,'inference_seconds':total_seconds,
                       'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'native_lm_head':True,'smoke':a.smoke})
        write_json(dest/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['by_source','rejected']},indent=2),flush=True)


if __name__=='__main__':
    main()
