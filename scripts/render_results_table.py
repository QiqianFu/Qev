#!/usr/bin/env python3
"""Render bilingual README tables from the recorded benchmark metrics."""
import json
import re
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from result_highlights import highlighted_models

ROOT = Path(__file__).resolve().parents[1]
QEV = {'Qev-9B', 'Qev-4B', 'Qev-2B', 'Qev-0.8B'}
MODELS = [
    ('jev', 'Jev'), ('qev_9b', 'Qev-9B'), ('kev_9b', 'Kev-9B'),
    ('qev_4b', 'Qev-4B'), ('jevany_4b_pointer', 'JevAny-4B Pointer'),
    ('qev_2b', 'Qev-2B'), ('qev_0p8b', 'Qev-0.8B'), ('nanojev_0p6b', 'NanoJev-0.6B'),
]
BENCHMARKS = [
    ('jevbench_public', 'JevBench public · 231', 'JevBench公开题 · 231题'),
    ('decision_dev_clean', 'Decision development · clean', 'decision_dev · clean'),
    ('transfer_dev_clean', 'Transfer development · clean', 'transfer_dev · clean'),
    ('mmlupro', 'MMLU-Pro · 1,000', 'MMLU-Pro · 1000题'),
    ('semif_handwritten', 'SemIf · 144 handwritten', 'SemIf · 144道手写题'),
    ('scienthoon', 'scienthoon · 873', 'scienthoon · 873题'),
    ('wanli', 'WANLI · 256', 'WANLI · 256题'),
    ('gsm8k', 'GSM8K · multiple choice', 'GSM8K · 选择题改编'),
    ('chessbench', 'ChessBench · 5,000', 'ChessBench · 5000题'),
    ('bpomp', 'BPoMP · variant mean', 'BPoMP · 变体平均'),
]


def results_table(chinese=False):
    results = json.loads((ROOT / 'results/benchmarks.json').read_text())['results']
    results.update(json.loads((ROOT / 'results/decision-index.json').read_text())['results'])
    headers = ['评测' if chinese else 'Benchmark'] + [name for _, name in MODELS]
    headers[1] = 'Jev（参考）' if chinese else 'Jev (reference)'
    rows = []
    for key, english, translated in BENCHMARKS:
        scores = {model: entry.get('accuracy', entry.get('raw'))
                  for model, entry in results[key]['models'].items()}
        cells = [translated if chinese else english]
        winners = highlighted_models(scores, [model for model, _ in MODELS])
        for model, _ in MODELS:
            value = f'{100 * scores[model]:.2f}' if model in scores else '—'
            if model in winners:
                value = f'**{value}**'
            cells.append(value)
        rows.append(cells)
    return headers, rows


def render(language):
    chinese = language == 'zh-CN'
    headers, rows = results_table(chinese)
    widths = [212] + [106] * (len(headers) - 1)
    header_height, row_height = 66, 44
    width, height = sum(widths), header_height + len(rows) * row_height
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((name for name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC']
                if name in available), None)
    if chinese and cjk is None:
        raise RuntimeError('Install a Noto Sans CJK font to render the Chinese table.')
    plt.rcParams.update({
        'font.family': ([cjk] if chinese else ['DejaVu Sans']) + ['sans-serif'],
        'svg.fonttype': 'none',
    })
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), width, height, facecolor='white'))
    x = 0
    for col, (name, cell_width) in enumerate(zip(headers, widths)):
        ours = name in QEV
        if ours:
            ax.add_patch(Rectangle((x, 0), cell_width, height, facecolor='#e7f2ff'))
        ax.add_patch(Rectangle((x, 0), cell_width, header_height,
                              facecolor='#cfe5ff' if ours else '#f1f5f9'))
        title = name.replace('NanoJev-', 'NanoJev\n').replace(' Pointer', '\nPointer')
        title = title.replace(' (reference)', '\n(reference)').replace('（参考）', '\n（参考）')
        ax.text(x + (12 if col == 0 else cell_width / 2), header_height / 2, title,
                ha='left' if col == 0 else 'center', va='center', fontsize=10.2,
                fontweight='bold', color='#18549a' if ours else '#334155', linespacing=1.4)
        for row, cells in enumerate(rows):
            value = cells[col]
            bold = value.startswith('**') and value.endswith('**')
            value = value.strip('*')
            cy = header_height + row * row_height + row_height / 2
            if col == 0:
                title, _, note = value.partition(' · ')
                ax.text(x + 12, cy - (7 if note else 0), title, ha='left', va='center',
                        fontsize=10.2, color='#334155')
                if note:
                    ax.text(x + 12, cy + 10, note, ha='left', va='center',
                            fontsize=8.2, color='#64748b')
            else:
                assert value == '—' or re.fullmatch(r'\d+\.\d{2}', value), value
                ax.text(x + cell_width / 2, cy, value, ha='center', va='center',
                        fontsize=11.5, fontweight='bold' if bold else 'normal', color='#26364a')
        x += cell_width
    for y in [header_height + i * row_height for i in range(len(rows) + 1)]:
        ax.plot([0, width], [y, y], color='#d9e2ee', linewidth=.6)
    name = 'evaluation-table' + ('.zh-CN' if chinese else '')
    buffer = StringIO()
    fig.savefig(buffer, format='svg', metadata={'Date': None, 'Title': 'Qev evaluation results',
                'Description': 'Qev-9B, Qev-4B, Qev-2B and Qev-0.8B columns have light blue backgrounds. A dash means not evaluated.'})
    (ROOT / 'assets' / f'{name}.svg').write_text(
        '\n'.join(line.rstrip() for line in buffer.getvalue().splitlines()) + '\n')
    preview = ROOT / 'runs/validation'
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f'{name}.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    for language in ['en', 'zh-CN']:
        render(language)
