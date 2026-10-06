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
from result_highlights import highlighted_models

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
COVER_ORDER = ['mmlupro', 'semif_handwritten', 'jevbench_public', 'scienthoon',
               'wanli', 'chessbench', 'bpomp']
COVER_PANELS = [next(panel for panel in PANELS if panel[0] == key) for key in COVER_ORDER]
BAR_MODELS = [
    ('qev_4b', 'Qev-4B', '#7652bb'),
    ('jevany_4b_pointer', 'JevAny-4B · Pointer', '#70a7b7'),
    ('kev_4b', 'Kev-4B', '#d3a15b'),
    ('jev', 'Jev · reference', '#94a5b9'),
]
MATRIX = [('jev', 'Jev · reference'), ('qev_9b', 'Qev-9B'),
          ('kev_9b', 'Kev-9B'),
          ('qev_4b', 'Qev-4B'), ('kev_4b', 'Kev-4B'),
          ('jevany_4b_pointer', 'JevAny-4B · Pointer'),
          ('jevany_4b_direct', 'JevAny-4B · Direct-Token'),
          ('qev_2b', 'Qev-2B'), ('qev_0p8b', 'Qev-0.8B'),
          ('nanojev_0p6b', 'NanoJev-0.6B')]


def score(entry, model):
    if model in entry.get('not_evaluated', {}):
        return np.nan
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
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    fig.subplots_adjust(left=.035, right=.997, bottom=.21, top=.74)
    fig.text(.014, .951, 'Qev · 4B model comparison', fontsize=19, fontweight='semibold', color='#35445e')
    fig.text(.014, .885, 'Scores (%) · Qev, JevAny and Kev, with Jev as a hosted reference', fontsize=11.5, color='#7c89a0')
    ax.legend(handles=[Patch(facecolor=c, label=n) for _,n,c in BAR_MODELS], loc='lower left',
              bbox_to_anchor=(-.007, 1.025), ncol=4, frameon=False, labelcolor='#46546b', fontsize=11.5,
              handlelength=1.6, columnspacing=1.4)
    for i, (key, _, _) in enumerate(COVER_PANELS):
        row = results[key]
        values = [100*score(row, model) for model, _, _ in BAR_MODELS]
        for j, (model, _, color) in enumerate(BAR_MODELS):
            value = values[j]
            x = i + (j - (len(BAR_MODELS)-1)/2) * .225
            ax.bar(x, value, width=.195, color=color, linewidth=0, zorder=3)
            ax.text(x, value+2, f'{value:.1f}', ha='center', va='bottom', fontsize=11,
                    color='#40516a')
    ax.set_ylim(0, 108); ax.set_xlim(-.49, len(COVER_PANELS)-.51); ax.set_xticks([])
    ax.set_yticks([0,20,40,60,80,100])
    ax.tick_params(axis='y', length=0, pad=4, labelcolor='#8190a6', labelsize=10.5)
    ax.yaxis.grid(True, color='#e9edf4', linewidth=.8, zorder=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name=='bottom'); spine.set_color('#d7dfea')
    for i, (_, title, count) in enumerate(COVER_PANELS):
        ax.text(i, -.07, title, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=12, color='#35445e')
        ax.text(i, -.16, count, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10.5, color='#8190a6')
    fig.text(.014, .035, 'Qev: BF16 backbone. Kev / JevAny: FP32. ChessBench / BPoMP: Decision Index raw scores.', fontsize=10, color='#8190a6')
    save(fig, 'evaluation')

    values = np.array([[100*score(results[key], model) for key,_,_ in PANELS] for model,_ in MATRIX])
    fig, ax = plt.subplots(figsize=(16.8, 7.7))
    fig.subplots_adjust(left=.205, right=.983, top=.84, bottom=.13)
    fig.text(.038, .95, 'Qev · benchmark matrix', fontsize=17, fontweight='semibold', color='#35445e')
    fig.text(.038, .902, 'Scores (%) · Qev and reference models · — means not evaluated', fontsize=10.5, color='#7c89a0')
    cmap = plt.colormaps['Purples'].with_extremes(bad='#f1f5f9')
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(PANELS)), [title for _,title,_ in PANELS], fontsize=9.5, color='#35445e')
    ax.set_yticks(range(len(MATRIX)), [name for _,name in MATRIX], fontsize=10.5, color='#35445e')
    ax.tick_params(length=0, pad=9)
    winners = {key: highlighted_models(
        {model: score(results[key], model) for model, _ in MATRIX if model in results[key]['models']},
        [model for model, _ in MATRIX]) for key, _, _ in PANELS}
    for row, (model, _) in enumerate(MATRIX):
        for col, (key, _, _) in enumerate(PANELS):
            bold = model in winners[key]
            label = f'{values[row,col]:.2f}' if np.isfinite(values[row,col]) else '—'
            ax.text(col,row,label,ha='center',va='center',fontsize=11,
                    color='white' if values[row,col]>=65 else '#35445e',fontweight='bold' if bold else 'normal')
    ax.set_xticks(np.arange(-.5, len(PANELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MATRIX), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.038, .035, 'Development: clean. SemIf: handwritten. New tasks: Decision Index raw scores (BPoMP mean across variants).', fontsize=9, color='#8190a6')
    save(fig, 'evaluation-matrix')


if __name__ == '__main__':
    main()
