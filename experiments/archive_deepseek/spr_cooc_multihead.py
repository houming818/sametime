"""
多头: 共现表选 path → 词级 BLEU-1
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, re
import sentencepiece as spm
from collections import defaultdict, Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')

# ── ECDICT ──
with open('/workspace/multi_sense_anchors.json') as f: dict_data = json.load(f)

en_to_zhs = {}  # EN → {zh1, zh2, ...}
en_first_zh = {}  # EN → first sense
for en, senses in dict_data.items():
    for zh, pos in senses:
        ei = sp.encode_as_ids(en); zi = sp.encode_as_ids(zh)
        if all(x != 0 for x in ei) and all(x != 0 for x in zi) and 2 <= len(en) <= 15 and pos in ('n','v','a','adj','adv','vi','vt') and 1 <= len(zh) <= 6:
            en_to_zhs.setdefault(en, list()).append(zh)
            if en not in en_first_zh:
                en_first_zh[en] = zh

multi_words = {en: zhs for en, zhs in en_to_zhs.items() if len(set(zhs)) >= 2}
print(f"ECDICT: {len(en_first_zh)} words, {len(multi_words)} multi-sense")

# ── 构建共现表 ──
W = 3
cooc = defaultdict(Counter)
print("Building co-occurrence table...")
count = 0
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if count >= 80000: break
        if '\t' not in l: continue
        en, zh = l.strip().split('\t', 1)
        en = en.strip().lower()
        en_words = en.split()
        
        for i, w in enumerate(en_words[:60]):
            wc = w.strip('.,!?;:()[]"')
            if wc not in multi_words: continue
            
            matched = None
            for zh_s in multi_words[wc]:
                if zh_s in zh:
                    matched = zh_s; break
            if not matched: continue
            
            ctx_start = max(0, i - W)
            ctx_end = min(len(en_words), i + W + 1)
            for j in range(ctx_start, ctx_end):
                if j == i: continue
                cw = en_words[j].strip('.,!?;:()[]"')
                if len(cw) >= 2:
                    cooc[(wc, cw)][matched] += 1
        count += 1
        if count % 20000 == 0:
            print(f"  {count} sentences, {len(cooc)} entries")

print(f"Co-occurrence entries: {len(cooc)}")
for w in ['cold', 'light', 'spring', 'right', 'hard', 'fire']:
    entries = [(k, v) for k, v in cooc.items() if k[0] == w]
    if entries:
        print(f"  {w}:")
        for (ew, cw), zhs in sorted(entries, key=lambda x: -sum(x[1].values()))[:3]:
            print(f"    ({cw:15s}) → {dict(zhs.most_common(2))}")

# ── 翻译 ──
def predict(en_word, context_words, mode='single'):
    if mode == 'single' or en_word not in multi_words:
        return en_first_zh.get(en_word, None)
    
    scores = Counter()
    for cw in context_words:
        if (en_word, cw) in cooc:
            for zh, cnt in cooc[(en_word, cw)].items():
                scores[zh] += cnt
    if scores:
        return scores.most_common(1)[0][0]
    return en_first_zh.get(en_word, None)

test_sents = [
    ("the sun is hot today",              "太阳很热今天"),
    ("i love to eat bread",               "我爱吃面包"),
    ("he drinks milk every day",          "他每天喝牛奶"),
    ("the river is long and deep",        "河又长又深"),
    ("she gave me a red flower",          "她给了我一朵红花"),
    ("we walk to the big mountain",       "我们走到大山"),
    ("they live in a new house",          "他们住在新房子里"),
    ("my mother bought rice and fish",    "我母亲买了米和鱼"),
    ("the old man died last night",       "老人昨晚死了"),
    ("i want to see the moon and stars",  "我想看月亮和星星"),
    ("fire is hot and water is cold",     "火是热的水是冷的"),
    ("the bird flies in the sky",         "鸟在天上飞"),
    ("he cut the bread with a knife",     "他用刀切面包"),
    ("she has a strong heart",            "她有一颗坚强的心"),
    ("we drink tea every morning",        "我们每天早上喝茶"),
    ("it is a good day",                  "今天是个好天"),
    ("they kill the big fish",            "他们杀大鱼"),
    ("come and see the gold star",        "来看金星"),
    ("the new year is here",              "新年到了"),
    ("love is strong and true",           "爱是坚强而真实的"),
]

total, correct_s, correct_m, correct_o = 0, 0, 0, 0
for en_s, zh_ref in test_sents:
    en_words = en_s.split()
    for i, w in enumerate(en_words):
        wc = w.strip('.,!?;:()[]"')
        if wc not in en_first_zh: continue
        total += 1
        
        ctx = [en_words[j].strip('.,!?;:()[]"') for j in range(max(0,i-W), min(len(en_words),i+W+1)) if j != i]
        
        s = predict(wc, ctx, 'single')
        m = predict(wc, ctx, 'multi')
        
        if s and s in zh_ref: correct_s += 1
        if m and m in zh_ref: correct_m += 1
        # Oracle
        if wc in multi_words:
            if any(z in zh_ref for z in multi_words[wc]):
                correct_o += 1
        else:
            if s and s in zh_ref: correct_o += 1

print(f"\n=== BLEU-1 ===")
print(f"  Dict words: {total}")
print(f"  Single (first): {correct_s} = {100*correct_s/total:.1f}%")
print(f"  Multi  (cooc):  {correct_m} = {100*correct_m/total:.1f}%")
print(f"  Oracle (any):   {correct_o} = {100*correct_o/total:.1f}%")
print(f"  Δ multi-single = {100*correct_m/total - 100*correct_s/total:+.1f}%")
print("Done.")
