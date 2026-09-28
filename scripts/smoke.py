#!/usr/bin/env python3
"""Offline CPU integration check using a tiny random hybrid Qwen3.5 model."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from qev.encoding import SPECIAL, Limits
    from qev.model import ModelSpec
    from qev.prepare import build_dataset
    torch.set_num_threads(1)
    torch.manual_seed(17)
    symbols = ['<pad>', '<unk>', *SPECIAL, *sorted(pre_tokenizers.ByteLevel.alphabet())]
    backend = Tokenizer(models.BPE({s:i for i,s in enumerate(symbols)}, [], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>',
                                       additional_special_tokens=list(SPECIAL))
    config = Qwen3_5TextConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=1024, attention_dropout=0.0, pad_token_id=0,
        layer_types=['linear_attention','full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':1.0,
                         'mrope_section':[2,3,3],'mrope_interleaved':True})
    config._attn_implementation = 'eager'
    base = out/'base'
    Qwen3_5TextModel(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = build_dataset(ROOT/'examples/train.jsonl', out/'data', validation=ROOT/'examples/dev.jsonl')
    spec = ModelSpec(str(base), head_dim=32, head_heads=4, head_layers=2, lora_rank=2,
                     weights_dtype='fp32', attention='eager', candidate_interaction='last-full-attention')
    settings = {'epochs':1, 'batch_size':1, 'accum':1, 'seed':17, 'lr':1e-3, 'head_lr':1e-3,
        'joint_gate_lr':1e-3, 'head_warmup_steps':0, 'warmup_steps':0, 'save_every':1,
        'gradient_checkpointing':False, 'autocast':'fp32', 'prefix_execution':'tree-batched'}
    cfg = out/'config.json'
    cfg.write_text(json.dumps({'model':asdict(spec),'limits':asdict(Limits(256,256,128,640,32)),
                              'training':settings},indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    # The source checkout is explicit; no original research directory is imported.
    env['PYTHONPATH'] = str(ROOT)
    def run(module, *args):
        subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True,cwd=ROOT,env=env)
    train = ['--config',cfg,'--data',data,'--out',out/'train','--device','cpu']
    run('qev.train',*train,'--max-steps','1')
    run('qev.train',*train,'--resume',out/'train/step-000001','--max-steps','2')
    checkpoint=out/'train/step-000002'
    run('qev.predict','--checkpoint',checkpoint,'--input',ROOT/'examples/requests.jsonl',
        '--out',out/'predictions.jsonl','--device','cpu','--reference')
    run('qev.evaluate','--checkpoint',checkpoint,'--data',data,'--split','dev',
        '--out',out/'evaluation','--device','cpu','--reference')
    rows=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==1 and len(rows[0]['questions'])==3
    report=json.loads((out/'evaluation/report.json').read_text())
    assert report['answered_questions']==6 and report['rejected_questions']==0
    proof={'passed':True,'scope':'tiny random hybrid Qwen3.5; CPU; no benchmark quality claim',
           'steps':2,'prediction_questions':3,'evaluation_questions':6}
    (out/'smoke.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
