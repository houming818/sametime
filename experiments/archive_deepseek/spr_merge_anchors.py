"""Merge all pos-aligned pairs into spr_anchor_bridge.py ANCHOR_WORDS."""
import sentencepiece as spm, re
from collections import defaultdict, Counter

sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')

pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 2 and len(en_t) >= 2: pairs.append((zh_t, en_t))

en_zh = defaultdict(Counter)
for zh, en in pairs[:100000]:
    Te, Tz = len(en), len(zh)
    for t in range(Te):
        z_pos = int(t * Tz / Te)
        en_zh[en[t]][zh[z_pos]] += 1
    for t in range(Tz):
        e_pos = int(t * Te / Tz)
        en_zh[en[e_pos]][zh[t]] += 1

all_pairs = set()
for e_tok, zh_counts in en_zh.items():
    total = sum(zh_counts.values())
    if total < 3: continue
    for z_tok, count in zh_counts.most_common(2):
        if count < 10: continue  # medium threshold for more anchors
        ep = sp.id_to_piece(e_tok).replace('\u2581', '')
        zp = sp.id_to_piece(z_tok).replace('\u2581', '')
        if len(ep) < 1 or len(zp) < 1: continue
        if ep.isdigit() or zp.isdigit(): continue
        all_pairs.add((ep, zp))

# Also load existing manual pairs from ANCHOR_WORDS
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    idx = txt.find("ANCHOR_WORDS = [")
    idx2 = txt.find("]", idx)
    block = txt[idx:idx2+1]
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", block):
        all_pairs.add((m.group(1), m.group(2)))

print(f"Total pairs: {len(all_pairs)}")

# Write ALL_PAIRS as a Python list
with open('/workspace/all_anchors.py', 'w') as f:
    f.write(f"# Auto-generated anchor pairs from parallel corpus ({len(all_pairs)} pairs)\n")
    f.write("ALL_ANCHORS = [\n")
    for ep, zp in sorted(all_pairs):
        f.write(f"    ({repr(ep)}, {repr(zp)}),\n")
    f.write("]\n")
print("Written to /workspace/all_anchors.py")
