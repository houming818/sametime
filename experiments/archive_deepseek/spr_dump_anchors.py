"""Extract MAXIMUM anchor pairs with very loose thresholds."""
import sentencepiece as spm, re
from collections import defaultdict, Counter

sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()

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
        e_tok = en[t]; z_pos = int(t * Tz / Te); z_tok = zh[z_pos]
        en_zh[e_tok][z_tok] += 1
    for t in range(Tz):
        z_tok = zh[t]; e_pos = int(t * Te / Tz); e_tok = en[e_pos]
        en_zh[e_tok][z_tok] += 1

# Very loose thresholds
all_pairs = set()
for e_tok, zh_counts in en_zh.items():
    total = sum(zh_counts.values())
    if total < 3: continue
    for z_tok, count in zh_counts.most_common(2):
        if count < 2: continue
        conf = count / total
        ep = sp.id_to_piece(e_tok).replace('▁', '')
        zp = sp.id_to_piece(z_tok).replace('▁', '')
        if len(ep) < 1 or len(zp) < 1: continue
        if ep.isdigit() or zp.isdigit(): continue
        all_pairs.add((ep, zp))

# Also include the existing 694 manual pairs
manual = []
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    idx = txt.find("ANCHOR_WORDS = [")
    idx2 = txt.find("]", idx)
    block = txt[idx:idx2+1]
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", block):
        manual.append((m.group(1), m.group(2)))
print(f"Manual: {len(manual)}")

# Merge and deduplicate
all_pairs.update(manual)
print(f"Total unique: {len(all_pairs)}")

# Print as Python list
print("\nALL_ANCHORS = [")
for ep, zp in sorted(all_pairs):
    # Escape single quotes in strings
    ep_s = ep.replace("'", "\\'")
    zp_s = zp.replace("'", "\\'")
    print(f"    ('{ep_s}', '{zp_s}'),")
print("]")
