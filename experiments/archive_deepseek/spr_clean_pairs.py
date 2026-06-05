import sentencepiece as spm, re
from collections import defaultdict, Counter

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')

pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip())
            en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3:
                pairs.append((zh_t, en_t))

en_zh = defaultdict(Counter)
for zh, en in pairs[:80000]:
    Te, Tz = len(en), len(zh)
    for t in range(Te):
        e_tok = en[t]; z_pos = int(t * Tz / Te); z_tok = zh[z_pos]
        en_zh[e_tok][z_tok] += 1
    for t in range(Tz):
        z_tok = zh[t]; e_pos = int(t * Te / Tz); e_tok = en[e_pos]
        en_zh[e_tok][z_tok] += 1

all_pairs = set()
for e_tok, zh_counts in en_zh.items():
    total = sum(zh_counts.values())
    if total < 5: continue
    for z_tok, count in zh_counts.most_common(1):
        if count < 5: continue
        conf = count / total
        if conf < 0.20: continue
        ep = sp.id_to_piece(e_tok).replace('▁', '')
        zp = sp.id_to_piece(z_tok).replace('▁', '')
        if len(ep) < 2 or len(zp) < 2: continue
        if re.match(r'^[\d\W_]+$', ep) or re.match(r'^[\d\W_]+$', zp): continue
        if re.match(r'^[a-z]{1,3}$', ep): continue
        if ep.isdigit() or zp.isdigit(): continue
        all_pairs.add((ep, zp, count, int(conf * 100)))

sorted_pairs = sorted(all_pairs, key=lambda x: -x[2])
print(f"Clean pairs: {len(sorted_pairs)}")

print("\n# === ALL_CLEAN_ANCHORS ===")
print("[")
for ep, zp, count, conf in sorted_pairs:
    print(f"    ('{ep}', '{zp}'),  # c={count}")
print("]")
