# SPDX-License-Identifier: Apache-2.0
"""Continue a student with teacher probabilities and representation responses."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import torch

from .checkpoint import load_model, save_model, rng_state, restore_rng
from .data import file_hash, write_json
from .refinement import distribution_loss, refinement_plan, microbatches
from .representation_response import CaptureTaskStates, response_geometry_loss
from .response_data import load_response_data
from .schema import Record


def learning_rate_scale(step, total, warmup, minimum=.1):
    if not 0 <= step < total or not 0 <= minimum <= 1 or warmup < 0:
        raise ValueError('invalid learning-rate schedule')
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def train(checkpoint, data, config, out, *, device='cuda', resume=None, stop_after=None):
    cfg = dict(config)
    for name in ('steps', 'batch_endpoints', 'microbatch_endpoints'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not 2 <= cfg['microbatch_endpoints'] <= cfg['batch_endpoints']:
        raise ValueError('microbatch must fit a pair and not exceed the full batch')
    if not 0 <= cfg['response_weight'] or not math.isfinite(cfg['response_weight']):
        raise ValueError('invalid response weight')
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(out)
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    rows, manifest = load_response_data(data)
    data = Path(data)
    # Automatic resume protection; no caller-supplied checksums or archive service.
    identity = {name: file_hash(data / info['file']) for name, info in manifest['files'].items()}
    identity['manifest'] = file_hash(data / 'manifest.json')
    model, tokenizer, encoder, _ = load_model(resume or checkpoint, device)
    if model.spec.lora_dropout or model.temperature != 1 or model.lora_merged:
        raise ValueError('distillation requires zero dropout, temperature 1 and an unmerged adapter')
    if device == 'cpu' and model.spec.weights_dtype != 'fp32':
        raise ValueError('CPU training requires an FP32 checkpoint')
    model.prefix_execution = 'tree-batched'
    model.tree_checkpointing = cfg.get('gradient_checkpointing', True)
    model.backbone.config.use_cache = False
    model.head_only = False
    pairs = [(encoder(Record.from_dict(row['original'])), encoder(Record.from_dict(row['edited'])))
             for row in rows['pairs']]
    replay = [encoder(Record.from_dict(row['record'])) for row in rows['replay']]
    plans = refinement_plan(len(pairs), [r.record.source for r in replay], cfg['steps'],
                            cfg['batch_endpoints'], cfg['replay_fraction'], cfg['seed'])
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if not backbone:
        raise ValueError('checkpoint has no trainable adapter')
    groups = [{'params': backbone, 'lr': cfg['lr']},
              {'params': list(model.head.parameters()), 'lr': cfg['head_lr']}]
    if model.joint_layer is not None:
        groups.append({'params': [model.joint_gate], 'lr': cfg['gate_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    base_lrs = [group['lr'] for group in groups]
    start = 0
    if resume:
        state = torch.load(Path(resume) / 'training.pt', map_location='cpu', weights_only=False)
        if (state.get('schema') != 'qev.response-training.v1' or state['config'] != cfg
                or state['data'] != identity):
            raise ValueError('resume configuration or training data changed')
        optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng']); start = state['step']
        if start >= cfg['steps']:
            raise ValueError('training already completed')
    if stop_after is not None and stop_after <= start:
        raise ValueError('stop-after must be greater than the saved step')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    last = None
    for index in range(start, cfg['steps']):
        model.train(); optimizer.zero_grad(set_to_none=True)
        scale = learning_rate_scale(index, cfg['steps'], cfg['warmup_steps'], cfg['minimum_lr_fraction'])
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * scale
        ce_value = response_value = 0.
        for units in microbatches(plans[index], cfg['microbatch_endpoints']):
            encoded = []
            for kind, i in units:
                encoded.extend(pairs[i] if kind == 'pair' else [replay[i]])
            context = torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
            with context, CaptureTaskStates(model.head) as captured:
                outputs = model(encoded)
                features = captured.features
                if len(features) != len(encoded) or any(len(output) != 1 for output in outputs):
                    raise ValueError('unaligned student head states')
                offset = 0
                terms, response_terms = [], []
                kwargs = {'teacher_temperature': cfg['teacher_temperature'],
                          'student_temperature': cfg['student_temperature']}
                for kind, i in units:
                    if kind == 'replay':
                        terms.append(distribution_loss(outputs[offset][0], rows['replay'][i]['teacher_logits'], **kwargs))
                        offset += 1
                    else:
                        row = rows['pairs'][i]
                        terms.extend(distribution_loss(outputs[offset + j][0], row[key], **kwargs)
                                     for j, key in enumerate(('teacher_original_logits', 'teacher_edited_logits')))
                        response = response_geometry_loss(features[offset], features[offset+1],
                                                          row['response'], kind='anchored_response')
                        response_terms.append(2 * cfg['response_weight'] * response / manifest['response_scale'])
                        offset += 2
                point_loss = sum(terms) / cfg['batch_endpoints']
                response_loss = sum(response_terms, point_loss * 0) / cfg['batch_endpoints']
                loss = point_loss + response_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite distillation loss')
            loss.backward()
            ce_value += float(point_loss.detach()); response_value += float(response_loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step = index + 1
        log = {'step': step, 'loss': ce_value + response_value, 'cross_entropy': ce_value,
               'response_loss': response_value, 'gradient_norm': float(norm)}
        with (out / 'train.jsonl').open('a') as stream:
            stream.write(json.dumps(log) + '\n')
        if step == 1 or step % 10 == 0:
            print(json.dumps(log), flush=True)
        stop = stop_after is not None and step >= stop_after
        if step in cfg['save_steps'] or step == cfg['steps'] or stop:
            destination = out / f'step-{step:06d}'
            temporary = out / f'.step-{step:06d}.incomplete'
            save_model(temporary, model, tokenizer, encoder.limits, {'training': 'response-distillation'})
            torch.save({'schema': 'qev.response-training.v1', 'config': cfg, 'data': identity,
                        'step': step, 'optimizer': optimizer.state_dict(), 'rng': rng_state()},
                       temporary / 'training.pt')
            temporary.rename(destination)
            last = destination
            write_json(out / 'latest.json', {'checkpoint': destination.name, 'step': step})
        if stop:
            break
    write_json(out / 'outcome.json', {'completed': step == cfg['steps'], 'step': step,
                                     'examples_seen': step * cfg['batch_endpoints'], 'checkpoint': last.name})
    return last


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True, help='student after probability distillation')
    p.add_argument('--data', required=True, help='output of qev.teacher responses')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args()
    train(a.checkpoint, a.data, json.loads(Path(a.config).read_text()), a.out,
          device=a.device, resume=a.resume, stop_after=a.stop_after)


if __name__ == '__main__':
    main()
