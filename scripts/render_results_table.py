#!/usr/bin/env python3
"""Render the README score tables with blue backgrounds for Qev columns."""
import re
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
QEV = {'Qev-9B', 'Qev-4B', 'Qev-2B'}


def read_table(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(('| Benchmark |', '| 评测 |')))
    rows = []
    for line in lines[start:]:
        if not line.startswith('|'):
            break
        rows.append([cell.strip() for cell in line.strip('|').split('|')])
    return rows[0], rows[2:]


def render(language):
    chinese = language == 'zh-CN'
    headers, rows = read_table(ROOT / ('README.zh-CN.md' if chinese else 'README.md'))
    widths = [212] + [106] * (len(headers) - 1)
    header_height, row_height = 66, 44
    width, height = sum(widths), header_height + len(rows) * row_height
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((name for name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC']
                if name in available), None)
    if chinese and cjk is None:
        raise RuntimeError('Install a Noto Sans CJK font to render the Chinese table.')
    plt.rcParams.update({
        'font.family': ['DejaVu Sans'] + ([cjk] if chinese else []) + ['sans-serif'],
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
        title = name.replace('Qwen3.5-', 'Qwen3.5\n').replace(' Pointer', '\nPointer')
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
                assert re.fullmatch(r'\d+\.\d{2}', value), value
                ax.text(x + cell_width / 2, cy, value, ha='center', va='center',
                        fontsize=11.5, fontweight='bold' if bold else 'normal', color='#26364a')
        x += cell_width
    for y in [header_height + i * row_height for i in range(len(rows) + 1)]:
        ax.plot([0, width], [y, y], color='#d9e2ee', linewidth=.6)
    name = 'evaluation-table' + ('.zh-CN' if chinese else '')
    buffer = StringIO()
    fig.savefig(buffer, format='svg', metadata={'Date': None, 'Title': 'Qev evaluation results',
                'Description': 'Qev-9B, Qev-4B and Qev-2B columns have light blue backgrounds.'})
    (ROOT / 'assets' / f'{name}.svg').write_text(
        '\n'.join(line.rstrip() for line in buffer.getvalue().splitlines()) + '\n')
    preview = ROOT / 'runs/validation'
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f'{name}.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    for language in ['en', 'zh-CN']:
        render(language)
