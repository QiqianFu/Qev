#!/usr/bin/env python3
"""Regenerate the public JevBench figure from integer counts; requires matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'results/benchmarks.json').read_text())['results']['jevbench_public']
keys=['jev','qev_9b','kev_9b','qwen35_9b_base']
labels=['Jev 1.13.0','Qev-9B','Kev-9B (pinned)','Qwen3.5-9B-Base']
colors=['#239781','#7048e8','#9ba7ba','#c5cedb']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained')
values=[100*rows['models'][k]['accuracy'] for k in keys]
ax.barh(range(4),values,color=colors,height=.55)
for i,k in enumerate(keys):
 v=rows['models'][k];ax.text(values[i]+1,i,f"{values[i]:.2f}%  ·  {v['correct']}/{rows['total']}",va='center',fontsize=10,color='#363b50')
ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,110)
ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Accuracy (%) · 231 public questions')
ax.set_title('JevBench public · fixed model revisions',loc='left',fontweight='bold',pad=16,color='#252a40')
ax.xaxis.grid(True,color='#e8ebf1');ax.set_axisbelow(True)
for s in ax.spines.values():s.set_visible(False)
ax.tick_params(axis='both',length=0);ax.tick_params(axis='y',pad=10)
fig.savefig(ROOT/'assets/jevbench.svg',metadata={'Date':None})
fig.savefig(ROOT/'runs/validation/jevbench.png',dpi=130)

svg = ROOT/'assets/jevbench.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
