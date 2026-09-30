#!/usr/bin/env python3
"""Plot matched 9B and 2B comparisons and a six-model accuracy matrix."""
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
    ('decision_dev_clean', 'Decision dev', '1,264 clean'),
    ('transfer_dev_clean', 'Transfer dev', '656 clean'),
    ('mmlupro', 'MMLU-Pro', '1,000 questions'),
    ('semif_handwritten', 'SemIf', '144 handwritten'),
    ('scienthoon', 'scienthoon', '873 questions'),
    ('wanli', 'WANLI', '256 questions'),
    ('jevbench_public', 'JevBench', '231 public'),
]
GROUPS = [
    ('9B models', [('qev_9b', 'Qev-9B · BF16', '#7652bb'), ('qwen35_9b_base', 'Qwen3.5-9B-Base · BF16', '#94a5b9')]),
    ('2B models', [('qev_2b', 'Qev-2B · BF16', '#ae7ed6'), ('qwen35_2b_base', 'Qwen3.5-2B-Base · BF16', '#94a5b9')]),
]
MATRIX = [('jev', 'Jev · reference'), ('qev_9b', 'Qev-9B'),
          ('kev_9b', 'Kev-9B'), ('qwen35_9b_base', 'Qwen3.5-9B-Base'),
          ('qev_2b', 'Qev-2B'), ('qwen35_2b_base', 'Qwen3.5-2B-Base')]


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
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.5), sharex=True)
    fig.subplots_adjust(left=.072, right=.992, bottom=.15, top=.87, hspace=.39)
    fig.text(.072, .963, 'Qev · 9B and 2B', fontsize=18, fontweight='semibold', color='#35445e')
    fig.text(.072, .926, 'Decision accuracy on the same seven benchmark subsets', fontsize=11, color='#7c89a0')
    for group_index, (ax, (title, models)) in enumerate(zip(axes, GROUPS)):
        ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=12, fontweight='semibold', color='#35445e')
        ax.legend(handles=[Patch(facecolor=c, label=n) for _,n,c in models], loc='lower right',
                  bbox_to_anchor=(1, 1.02), ncol=2, frameon=False, labelcolor='#46546b', fontsize=10)
        for i, (key, _, _) in enumerate(PANELS):
            row = results[key]
            values = [100*row['models'][model]['correct']/row['total'] for model,_,_ in models]
            for j, (_,_,color) in enumerate(models):
                x = i + (-.185 if j == 0 else .185)
                ax.bar(x, values[j], width=.315, color=color, linewidth=0, zorder=3)
                ax.text(x, values[j]+2, f'{values[j]:.2f}', ha='center', va='bottom', fontsize=10.4,
                        color='#40516a', fontweight='bold' if group_index==0 and values[j]>values[1-j] else 'normal')
        ax.set_ylim(0, 108); ax.set_xlim(-.65, len(PANELS)-.35); ax.set_xticks([])
        ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel('Accuracy (%)', color='#7c89a0', fontsize=10)
        ax.tick_params(axis='y', length=0, pad=8, labelcolor='#8190a6', labelsize=9)
        ax.yaxis.grid(True, color='#e9edf4', linewidth=.8, zorder=0)
        for name, spine in ax.spines.items():
            spine.set_visible(name=='bottom'); spine.set_color('#d7dfea')
    for i, (_, title, count) in enumerate(PANELS):
        axes[-1].text(i, -.065, title, transform=axes[-1].get_xaxis_transform(), ha='center', va='top', fontsize=10.5, color='#35445e')
        axes[-1].text(i, -.148, count, transform=axes[-1].get_xaxis_transform(), ha='center', va='top', fontsize=9, color='#8190a6')
    fig.text(.072, .035, 'Qev uses BF16 backbone computation and an FP32 decision head. Bold labels compare Qev-9B with Qwen3.5-9B-Base.', fontsize=9, color='#8190a6')
    save(fig, 'evaluation')

    values = np.array([[100*results[key]['models'][model]['accuracy'] for key,_,_ in PANELS] for model,_ in MATRIX])
    fig, ax = plt.subplots(figsize=(12.8, 4.9))
    fig.subplots_adjust(left=.195, right=.983, top=.81, bottom=.145)
    fig.text(.038, .95, 'Qev · benchmark matrix', fontsize=17, fontweight='semibold', color='#35445e')
    fig.text(.038, .902, 'Accuracy (%) · identical question subsets across all six models', fontsize=10.5, color='#7c89a0')
    ax.imshow(values, cmap='Purples', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(PANELS)), [title for _,title,_ in PANELS], fontsize=9.5, color='#35445e')
    ax.set_yticks(range(len(MATRIX)), [name for _,name in MATRIX], fontsize=10.5, color='#35445e')
    ax.tick_params(length=0, pad=9)
    comparisons = {'qev_9b': 'kev_9b', 'kev_9b': 'qev_9b'}
    for row, (model, _) in enumerate(MATRIX):
        for col, (key, _, _) in enumerate(PANELS):
            other = comparisons.get(model)
            bold = other is not None and results[key]['models'][model]['accuracy'] > results[key]['models'][other]['accuracy']
            ax.text(col,row,f'{values[row,col]:.2f}',ha='center',va='center',fontsize=11,
                    color='white' if values[row,col]>=65 else '#35445e',fontweight='bold' if bold else 'normal')
    ax.set_xticks(np.arange(-.5, len(PANELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MATRIX), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.038, .035, 'Development sets: clean questions. SemIf: 144 handwritten questions. JevBench: 231 public questions.', fontsize=9, color='#8190a6')
    save(fig, 'evaluation-matrix')


if __name__ == '__main__':
    main()
