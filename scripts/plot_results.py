#!/usr/bin/env python3
"""Draw paired benchmark bars from frozen integer counts; requires matplotlib."""
import json
from io import StringIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
MODELS = [('qev_9b', 'Qev-9B · BF16 backbone', '#7652bb'),
          ('kev_9b', 'Kev-9B · FP32', '#37958b')]


def main():
    results = json.loads((ROOT/'results/benchmarks.json').read_text())['results']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(left=.073, right=.993, bottom=.235, top=.79)
    fig.text(.073, .94, 'Qev-9B', fontsize=15, fontweight='semibold', color='#35445e')
    fig.text(.073, .885, 'Decision accuracy · seven benchmark panels', fontsize=11, color='#7c89a0')
    fig.legend(handles=[Patch(facecolor=color, label=name) for _,name,color in MODELS],
               loc='upper right', bbox_to_anchor=(.99,.961), frameon=False,
               ncol=2, handlelength=1.1, handleheight=1.05, columnspacing=2.3,
               labelcolor='#46546b', fontsize=11)
    for i, (key, title, count) in enumerate(PANELS):
        row = results[key]
        values = [100*row['models'][model]['correct']/row['total'] for model,_,_ in MODELS]
        for j, (_,_,color) in enumerate(MODELS):
            x = i + (-.185 if j == 0 else .185)
            ax.bar(x, values[j], width=.315, color=color, linewidth=0, zorder=3)
            ax.text(x, values[j]+2.1, f'{values[j]:.2f}', ha='center', va='bottom',
                    fontsize=11.3, color='#40516a',
                    fontweight='bold' if values[j] > values[1-j] else 'normal')
        ax.text(i, -.080, title, transform=ax.get_xaxis_transform(), ha='center',
                va='top', fontsize=11, color='#35445e')
        ax.text(i, -.158, count, transform=ax.get_xaxis_transform(), ha='center',
                va='top', fontsize=10, color='#8190a6')
    ax.set_xlim(-.68, len(PANELS)-.32)
    ax.set_ylim(0, 105)
    ax.set_xticks([])
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylabel('Accuracy (%)', labelpad=12, color='#7c89a0', fontsize=11)
    ax.tick_params(axis='y', length=0, pad=8, labelcolor='#8190a6', labelsize=10)
    ax.yaxis.grid(True, color='#e9edf4', linewidth=.8, zorder=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name == 'bottom')
        spine.set_color('#d7dfea')
    fig.text(.073, .057, 'Bold labels compare Qev with Kev. Jev and Qwen-base references are included in the detailed table.',
             fontsize=10, color='#8190a6')
    out = ROOT/'assets/evaluation.svg'
    buffer = StringIO()
    fig.savefig(buffer, format='svg', metadata={'Date':None})
    out.write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines())+'\n')
    preview = ROOT/'runs/validation'
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview/'evaluation-paired.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
