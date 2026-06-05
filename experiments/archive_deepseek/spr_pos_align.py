"""Position-approximate alignment: map parallel sentence tokens by relative position."""
import sentencepiece as spm
from collections import defaultdict, Counter

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
print(f"vocab={V}")

pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip())
            en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3:
                pairs.append((zh_t, en_t))
print(f"pairs={len(pairs)}")

# Position-approximate alignment
en_cooc = defaultdict(Counter)
for zh, en in pairs[:80000]:
    Te, Tz = len(en), len(zh)
    for t in range(Te):
        e_tok = en[t]
        z_pos = int(t * Tz / Te)  # relative position mapping
        z_tok = zh[z_pos]
        en_cooc[e_tok][z_tok] += 1

# Filter high-confidence pairs
MIN_COOC = 5
MIN_CONF = 0.3
good = []
for e_tok, zh_counts in en_cooc.items():
    total = sum(zh_counts.values())
    if total < 10: continue
    top_z, top_c = zh_counts.most_common(1)[0]
    if top_c < MIN_COOC: continue
    conf = top_c / total
    if conf < MIN_CONF: continue
    ep = sp.id_to_piece(e_tok)
    zp = sp.id_to_piece(top_z)
    if len(ep) < 2 or len(zp) < 2: continue
    good.append((ep, zp, top_c, conf, e_tok, top_z))

good.sort(key=lambda x: -x[2])
print(f"Good pairs: {len(good)}")

print("\n# === POS_ALIGNED_ANCHORS ===")
print("[")
for ep, zp, c, conf, e_tok, z_tok in good[:200]:
    print(f"    ('{ep}', '{zp}'),  # count={c} conf={conf:.2f}")
print("]")

# Show examples with sentences
print(f"\n# === Sample alignments ===")
for i, (ep, zp, c, conf, e_tok, z_tok) in enumerate(good[:15]):
    print(f"\n--- {i+1}: {ep} → {zp} (count={c}, conf={conf:.2f}) ---")
    shown = 0
    for zh, en in pairs[:80000]:
        if e_tok in en and z_tok in zh:
            en_text = sp.decode_ids(en)[:80]
            zh_text = sp.decode_ids(zh)[:80]
            print(f"  EN: {en_text}")
            print(f"  ZH: {zh_text}")
            shown += 1
            if shown >= 2: break
