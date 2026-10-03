#!/usr/bin/env python3
"""Plot the 4B comparison and the full benchmark matrix from recorded scores."""
import json
from io import StringIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANELS = [
    ('jevbench_public', 'JevBench', '231 public'),
    ('decision_dev_clean', 'Decision dev', '1,264 clean'),
    ('transfer_dev_clean', 'Transfer dev', '656 clean'),
    ('mmlupro', 'MMLU-Pro', '1,000 questions'),
    ('semif_handwritten', 'SemIf', '144 handwritten'),
    ('scienthoon', 'scienthoon', '873 questions'),
    ('wanli', 'WANLI', '256 questions'),
    ('gsm8k', 'GSM8K', '4 / 10 choices'),
    ('chessbench', 'ChessBench', '5,000 positions'),
    ('bpomp', 'BPoMP', 'variant mean'),
]
COVER_PANELS = [panel for panel in PANELS if panel[0] != 'decision_dev_clean']
BAR_MODELS = [
    ('qev_4b', 'Qev-4B', '#7652bb'),
    ('jevany_4b_pointer', 'JevAny-4B · Pointer', '#70a7b7'),
    ('kev_4b', 'Kev-4B', '#d3a15b'),
    ('jev', 'Jev · reference', '#94a5b9'),
]
MATRIX = [('jev', 'Jev · reference'), ('qev_9b', 'Qev-9B'),
          ('kev_9b', 'Kev-9B'), ('qwen35_9b_base', 'Qwen3.5-9B-Base'),
          ('qev_4b', 'Qev-4B'), ('kev_4b', 'Kev-4B'),
          ('jevany_4b_pointer', 'JevAny-4B · Pointer'),
          ('jevany_4b_direct', 'JevAny-4B · Direct-Token'),
          ('qwen35_4b_base', 'Qwen3.5-4B-Base'),
          ('qev_2b', 'Qev-2B'), ('qwen35_2b_base', 'Qwen3.5-2B-Base')]


def score(entry, model):
    row = entry['models'][model]
    return row['accuracy'] if 'accuracy' in row else row['raw']


def save(fig, name):
    buffer = StringIO()
    fig.savefig(buffer, format='svg', metadata={'Date': None})
    (ROOT/'assets'/f'{name}.svg').write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines())+'\n')
    preview = ROOT/'runs/validation'
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview/f'{name}.png', dpi=140)
    plt.close(fig)


def main():
    results = json.loads((ROOT/'results/benchmarks.json').read_text())['results']
    results.update(json.loads((ROOT/'results/decision-index.json').read_text())['results'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16.8, 6.0))
    fig.subplots_adjust(left=.065, right=.992, bottom=.21, top=.74)
    fig.text(.065, .951, 'Qev · 4B model comparison', fontsize=18, fontweight='semibold', color='#35445e')
    fig.text(.065, .885, 'Qev, JevAny and Kev · with Jev as a hosted reference', fontsize=11, color='#7c89a0')
    ax.legend(handles=[Patch(facecolor=c, label=n) for _,n,c in BAR_MODELS], loc='lower left',
              bbox_to_anchor=(-.007, 1.025), ncol=4, frameon=False, labelcolor='#46546b', fontsize=10)
    for i, (key, _, _) in enumerate(COVER_PANELS):
        row = results[key]
        values = [100*score(row, model) for model, _, _ in BAR_MODELS]
        for j, (model, _, color) in enumerate(BAR_MODELS):
            value = values[j]
            x = i + (j - (len(BAR_MODELS)-1)/2) * .225
            ax.bar(x, value, width=.195, color=color, linewidth=0, zorder=3)
            ax.text(x, value+2, f'{value:.1f}', ha='center', va='bottom', fontsize=8.8,
                    color='#40516a')
    ax.set_ylim(0, 108); ax.set_xlim(-.58, len(COVER_PANELS)-.42); ax.set_xticks([])
    ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel('Score (%)', color='#7c89a0', fontsize=10)
    ax.tick_params(axis='y', length=0, pad=8, labelcolor='#8190a6', labelsize=9)
    ax.yaxis.grid(True, color='#e9edf4', linewidth=.8, zorder=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name=='bottom'); spine.set_color('#d7dfea')
    for i, (_, title, count) in enumerate(COVER_PANELS):
        ax.text(i, -.07, title, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10, color='#35445e')
        ax.text(i, -.16, count, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9, color='#8190a6')
    fig.text(.065, .035, 'Qev: BF16 backbone. Kev / JevAny: FP32. New benchmarks use Decision Index raw scores.', fontsize=9, color='#8190a6')
    save(fig, 'evaluation')

    values = np.array([[100*score(results[key], model) for key,_,_ in PANELS] for model,_ in MATRIX])
    fig, ax = plt.subplots(figsize=(16.8, 7.7))
    fig.subplots_adjust(left=.205, right=.983, top=.84, bottom=.13)
    fig.text(.038, .95, 'Qev · benchmark matrix', fontsize=17, fontweight='semibold', color='#35445e')
    fig.text(.038, .902, 'Scores (%) · ten benchmarks across the Qev family, native bases and reference models', fontsize=10.5, color='#7c89a0')
    ax.imshow(values, cmap='Purples', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(PANELS)), [title for _,title,_ in PANELS], fontsize=9.5, color='#35445e')
    ax.set_yticks(range(len(MATRIX)), [name for _,name in MATRIX], fontsize=10.5, color='#35445e')
    ax.tick_params(length=0, pad=9)
    comparisons = {'qev_9b': 'kev_9b', 'kev_9b': 'qev_9b', 'qev_4b': 'kev_4b', 'kev_4b': 'qev_4b'}
    for row, (model, _) in enumerate(MATRIX):
        for col, (key, _, _) in enumerate(PANELS):
            other = comparisons.get(model)
            bold = other is not None and score(results[key], model) > score(results[key], other)
            ax.text(col,row,f'{values[row,col]:.2f}',ha='center',va='center',fontsize=11,
                    color='white' if values[row,col]>=65 else '#35445e',fontweight='bold' if bold else 'normal')
    ax.set_xticks(np.arange(-.5, len(PANELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MATRIX), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.038, .035, 'Development: clean. SemIf: handwritten. New tasks: Decision Index raw scores (BPoMP mean across variants).', fontsize=9, color='#8190a6')
    save(fig, 'evaluation-matrix')


if __name__ == '__main__':
    main()
