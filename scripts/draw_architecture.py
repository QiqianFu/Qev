#!/usr/bin/env python3
"""Render the bilingual Qev architecture overview as standalone vector graphics."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {
    'en': {
        'title': 'From a question to a decision',
        'subtitle': 'Understand each answer option. Compare its meaning. Return probabilities.',
        'tag': 'Qwen-based decisions',
        'context': 'SHARED CONTEXT', 'options': 'ANSWER OPTIONS',
        'option_note': 'Candidates = the answers you provide',
        'summaries': 'NUMERIC SUMMARIES', 'summary_note': 'Each option gets its own vector',
        'message': 'MESSAGE', 'quote': '“I was charged twice.”', 'question_label': 'QUESTION',
        'question': 'Which team should help?',
        'reuse1': 'The same message and question', 'reuse2': 'are used for every option.',
        'option': 'Option', 'names': ['Billing', 'Shipping', 'Account'],
        'summary_names': ['Billing summary', 'Shipping summary', 'Account summary'],
        'vector': 'A vector of 4,096 numbers', 'shared': 'shared model',
        'vector1': 'A vector is a list of numbers', 'vector2': 'that represents meaning.',
        'readouts': 'Readouts sent to the head',
        'readouts_note': 'Question summary h_q  +  option summaries e₁, e₂, e₃',
        'head': 'DECISION HEAD',
        'steps': ['Compact summaries', 'Compare options', 'Pair with question', 'Score each option'],
    },
    'zh-CN': {
        'title': '从问题到决策',
        'subtitle': '理解每个选项的含义，结合问题进行比较，输出选择概率。',
        'tag': '基于 Qwen 微调',
        'context': '共享上下文', 'options': '候选答案 / 选项',
        'option_note': 'Candidate 就是我们提供的备选答案',
        'summaries': '选项的数值摘要', 'summary_note': '每个选项对应一个摘要向量',
        'message': '输入消息', 'quote': '“同一订单被扣款两次。”', 'question_label': '问题',
        'question': '应该由哪个团队处理？',
        'reuse1': '同一份消息和问题', 'reuse2': '供每个选项共同使用。',
        'option': '选项', 'names': ['账单团队', '物流团队', '账户团队'],
        'summary_names': ['账单选项的摘要', '物流选项的摘要', '账户选项的摘要'],
        'vector': '向量 · 由 4,096 个数字组成', 'shared': '共享的模型',
        'vector1': '向量就是一组数字', 'vector2': '用来表示模型理解到的含义。',
        'readouts': '传入决策头的数值表示',
        'readouts_note': '问题摘要 h_q  +  三个选项摘要 e₁、e₂、e₃',
        'head': '决策头',
        'steps': ['压缩摘要', '比较选项', '结合问题', '逐项打分'],
    },
}

DEFS = '''<defs>
  <linearGradient id="background" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#f0f3fa"/><stop offset="1" stop-color="#e9eff5"/></linearGradient>
  <linearGradient id="face" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#fcfdff"/><stop offset="1" stop-color="#e9eef6"/></linearGradient>
  <linearGradient id="violet-face" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#f8f5ff"/><stop offset="1" stop-color="#e5def7"/></linearGradient>
  <linearGradient id="mint-face" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#f6fffc"/><stop offset="1" stop-color="#daeee8"/></linearGradient>
  <linearGradient id="mark"><stop stop-color="#7048e8"/><stop offset="1" stop-color="#3b2b8c"/></linearGradient>
  <linearGradient id="chip" x1="0%" y1="0%" x2="100%" y2="100%"><stop stop-color="#a891ed"/><stop offset="1" stop-color="#7048c6"/></linearGradient>
  <filter id="shadow" x="-25%" y="-40%" width="160%" height="200%"><feGaussianBlur stdDeviation="7"/></filter>
  <filter id="light" x="-25%" y="-40%" width="160%" height="200%"><feGaussianBlur stdDeviation="5"/></filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#9a9bb8" stroke-width="1.7" stroke-linejoin="round"/></marker>
</defs>'''


def render(lang):
    t = TEXT[lang]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="892" viewBox="0 0 1480 892" role="img" aria-labelledby="title description">',
             f'<title id="title">Qev · {escape(t["title"])}</title>',
             '<desc id="description">A shared message and question are combined with Billing, Shipping and Account answer options. Qwen computes a numeric summary vector e1, e2 or e3 for each option. Four decision-head stages turn summaries into scores.</desc>', DEFS,
             '<rect x="1" y="1" width="1478" height="890" rx="30" fill="url(#background)" stroke="#e2e7f0"/>']
    family = 'Noto Sans CJK SC,Segoe UI,Arial,sans-serif' if lang == 'zh-CN' else 'Inter,Segoe UI,Arial,sans-serif'
    parts.append(f'<g font-family="{family}">')

    def text(x, y, value, size=16, fill='#596781', weight=400, anchor='start', extra=''):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')

    def card(x, y, w, h, face='face', radius=21):
        parts.append(f'<rect x="{x+8}" y="{y+11}" width="{w}" height="{h}" rx="{radius}" fill="#8797af" opacity=".24" filter="url(#shadow)"/>')
        parts.append(f'<rect x="{x-3}" y="{y-3}" width="{w}" height="{h}" rx="{radius}" fill="#ffffff" opacity=".8" filter="url(#light)"/>')
        parts.append(f'<rect x="{x+1}" y="{y+4}" width="{w}" height="{h}" rx="{radius}" fill="#cfd6e5" opacity=".72"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="url(#{face})" stroke="#ffffff" stroke-opacity=".85" stroke-width="1.2"/>')

    def wire(d):
        parts.append(f'<path d="{d}" fill="none" stroke="#9a9bb8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"/>')

    # The wordmark uses the same gradient and letter shape as the README banner.
    text(59, 108, 'Qev', 83, 'url(#mark)', 800, extra='letter-spacing="-4" font-family="Inter,Segoe UI,Arial,sans-serif"')
    text(282, 69, t['title'], 31, '#29374f', 700)
    text(284, 104, t['subtitle'], 16, '#73809a')
    parts.append('<rect x="1215" y="47" width="199" height="43" rx="21" fill="#e8e2f8" stroke="#f9f7ff"/>')
    text(1314, 74, t['tag'], 14, '#71549d', 600, 'middle')
    parts.append('<path d="M64 150H1414" stroke="#dce3ef"/>')
    text(64, 191, t['context'], 13, '#8290a6', 700, extra='letter-spacing="1.2"')
    text(425, 191, t['options'], 13, '#498780', 700, extra='letter-spacing="1.2"')
    text(425, 217, t['option_note'], 13, '#7d8a9e')
    text(1000, 191, t['summaries'], 13, '#8266ab', 700, extra='letter-spacing="1.2"')
    text(1000, 217, t['summary_note'], 14, '#7d8a9e')

    # Draw connectors first so their ends sit behind the raised cards.
    wire('M350 412 H390 V290 H420')
    wire('M350 412 H420')
    wire('M390 412 V534 H420')
    for cy in (290, 412, 534):
        wire(f'M716 {cy} H792')
        wire(f'M946 {cy} H995')

    card(64, 310, 286, 210)
    text(88, 347, t['message'], 12, '#8290a6', 700, extra='letter-spacing="1"')
    text(88, 380, t['quote'], 19, '#35435b', 600)
    parts.append('<path d="M88 409H325" stroke="#dbe2ed"/>')
    text(88, 440, t['question_label'], 12, '#8290a6', 700, extra='letter-spacing="1"')
    text(88, 473, t['question'], 17, '#35435b', 600)
    text(64, 560, t['reuse1'], 14, '#8190a5')
    text(64, 583, t['reuse2'], 14, '#8190a5')

    for i, y in enumerate((244, 366, 488)):
        card(425, y, 288, 92, 'mint-face')
        parts.append(f'<rect x="443" y="{y+19}" width="53" height="53" rx="15" fill="#d1e9e1" stroke="#effaf5"/>')
        # Original, simple option icons: receipt, parcel and account.
        if i == 0:
            parts.append(f'<g transform="translate(457 {y+30})" stroke="#4c8d80" stroke-width="2" fill="none"><rect width="25" height="30" rx="4"/><path d="M6 9H19M6 15H19M6 21H14"/></g>')
        elif i == 1:
            parts.append(f'<g transform="translate(456 {y+31})" stroke="#4c8d80" stroke-width="2" fill="none" stroke-linejoin="round"><path d="M0 7L14 0L28 7V24L14 31L0 24ZM0 7L14 14L28 7M14 14V31M7 3.5L21 10.5"/></g>')
        else:
            parts.append(f'<g transform="translate(455 {y+30})" stroke="#4c8d80" stroke-width="2" fill="none"><circle cx="15" cy="8" r="7"/><path d="M2 31V27C2 15 28 15 28 27V31Z"/></g>')
        text(514, y+32, f'{t["option"]} {i+1}', 12, '#749189', 600)
        text(514, y+65, t['names'][i], 24, '#3d746c', 600)

        card(1000, y, 414, 92, 'violet-face')
        parts.append(f'<rect x="1020" y="{y+20}" width="57" height="57" rx="17" fill="#dcd2f1"/>')
        parts.append(f'<rect x="1018" y="{y+17}" width="57" height="57" rx="17" fill="#eee8fb" stroke="#ffffff"/>')
        text(1047, y+56, ['e₁','e₂','e₃'][i], 27, '#8061b1', 600, 'middle')
        text(1096, y+40, t['summary_names'][i], 20, '#655082', 600)
        text(1096, y+67, t['vector'], 14, '#9684aa')
        for j in range(3):
            for k in range(3):
                parts.append(f'<rect x="{1361+j*9}" y="{y+27+k*10}" width="5" height="6" rx="2" fill="#9b83c6" opacity="{.27+.13*((j+k+i)%3)}"/>')

    card(797, 244, 148, 336, 'violet-face', 25)
    # Three isometric slabs provide a visible 3D model symbol.
    for cy, tone in [(356, '#d0c4e8'), (337, '#bca8e4'), (318, 'url(#chip)')]:
        parts.append(f'<path d="M818 {cy}L871 {cy+28}L924 {cy}V{cy+13}L871 {cy+41}L818 {cy+13}Z" fill="#a38acc"/>')
        parts.append(f'<path d="M818 {cy}L871 {cy-28}L924 {cy}L871 {cy+28}Z" fill="{tone}" stroke="#f2ecff" stroke-width="1"/>')
        parts.append(f'<path d="M871 {cy+28}V{cy+41}L924 {cy+13}V{cy}Z" fill="#8566b2" opacity=".7"/>')
    text(871, 435, 'Qwen', 25, '#725397', 700, 'middle')
    text(871, 463, '+ LoRA', 15, '#9a83b5', 600, 'middle')
    text(871, 548, t['shared'], 13, '#9a83b5', 400, 'middle')

    text(64, 641, t['vector1'], 15, '#63798b', 600)
    text(64, 665, t['vector2'], 14, '#8190a5')
    parts.append('<rect x="425" y="615" width="989" height="64" rx="18" fill="#e5e3f2" stroke="#f9f8ff"/>')
    text(450, 640, t['readouts'], 13, '#8d7fa5', 600)
    text(450, 664, t['readouts_note'], 17, '#76608e', 500)
    text(64, 731, t['head'], 13, '#85729f', 700, extra='letter-spacing="1.7"')
    for i, (x, w) in enumerate([(64,299),(413,299),(762,299),(1111,303)]):
        card(x, 758, w, 85, 'mint-face' if i == 3 else 'face')
        text(x+w/2, 809, t['steps'][i], 20, '#467f75' if i == 3 else '#62718a', 600, 'middle')
        if i < 3:
            wire(f'M{x+w+7} 800 H{x+w+42}')
    parts.append('</g></svg>')
    return '\n'.join(parts)+'\n'


def main():
    for lang in TEXT:
        name = 'architecture.svg' if lang == 'en' else 'architecture.zh-CN.svg'
        (ROOT/'assets'/name).write_text(render(lang), encoding='utf-8')


if __name__ == '__main__':
    main()
