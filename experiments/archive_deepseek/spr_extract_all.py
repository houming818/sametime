"""Extract maximum anchor pairs from parallel corpus — position-aligned + bidirectional + co-occurrence."""
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

# ─── Method 1: Position-aligned (EN→ZH + ZH→EN) ───
en_to_zh = defaultdict(Counter)
zh_to_en = defaultdict(Counter)

for zh, en in pairs[:80000]:
    Te, Tz = len(en), len(zh)
    # EN→ZH
    for t in range(Te):
        e_tok = en[t]
        z_pos = int(t * Tz / Te)
        z_tok = zh[z_pos]
        en_to_zh[e_tok][z_tok] += 1
    # ZH→EN (reverse direction)
    for t in range(Tz):
        z_tok = zh[t]
        e_pos = int(t * Te / Tz)
        e_tok = en[e_pos]
        zh_to_en[z_tok][e_tok] += 1

def extract_aligned(coc_map, min_count=5, min_conf=0.15, min_total=10):
    good = []
    for src_tok, tgt_counts in coc_map.items():
        total = sum(tgt_counts.values())
        if total < min_total: continue
        top_tgt, top_c = tgt_counts.most_common(1)[0]
        if top_c < min_count: continue
        conf = top_c / total
        if conf < min_conf: continue
        sp_src = sp.id_to_piece(src_tok)
        sp_tgt = sp.id_to_piece(top_tgt)
        if len(sp_src) < 2 or len(sp_tgt) < 2: continue
        good.append((sp_src, sp_tgt, top_c, conf, src_tok, top_tgt))
    return good

# Tier 1: High confidence (conf >= 0.40)
print("\n=== Tier 1: conf >= 0.40 ===")
t1 = extract_aligned(en_to_zh, min_count=3, min_conf=0.40, min_total=5)
print(f"  EN→ZH: {len(t1)}")
t1b = extract_aligned(zh_to_en, min_count=3, min_conf=0.40, min_total=5)
print(f"  ZH→EN: {len(t1b)}")
t1_combined = set()
for s, t, c, conf, *_ in t1 + t1b:
    t1_combined.add((s, t))
print(f"  Combined unique: {len(t1_combined)}")

# Tier 2: Medium confidence (0.25 <= conf < 0.40)
print("\n=== Tier 2: 0.25 <= conf < 0.40 ===")
t2 = extract_aligned(en_to_zh, min_count=5, min_conf=0.25, min_total=10)
print(f"  EN→ZH: {len(t2)}")
t2b = extract_aligned(zh_to_en, min_count=5, min_conf=0.25, min_total=10)
print(f"  ZH→EN: {len(t2b)}")
t2_combined = set()
for s, t, c, conf, *_ in t2 + t2b:
    if (s, t) not in t1_combined:
        t2_combined.add((s, t))
print(f"  Combined unique (excl T1): {len(t2_combined)}")

# Tier 3: Low but useful (0.15 <= conf < 0.25)
print("\n=== Tier 3: 0.15 <= conf < 0.25 (bonus) ===")
t3 = extract_aligned(en_to_zh, min_count=8, min_conf=0.15, min_total=15)
print(f"  EN→ZH: {len(t3)}")
t3b = extract_aligned(zh_to_en, min_count=8, min_conf=0.15, min_total=15)
print(f"  ZH→EN: {len(t3b)}")
existing = t1_combined | t2_combined
t3_combined = set()
for s, t, c, conf, *_ in t3 + t3b:
    if (s, t) not in existing:
        t3_combined.add((s, t))
print(f"  Combined unique (excl T1+T2): {len(t3_combined)}")

# Print all tiers as ANCHOR_WORDS
print(f"\n# === Tier 1 ({len(t1_combined)} pairs) ===")
print("TIER1_ANCHORS = [")
for s, t in sorted(t1_combined | t2_combined | t3_combined, key=lambda x: x[0]):
    # Clean BPE prefixes for ANCHOR_WORDS format
    sc = s.replace('▁', '').strip()
    tc = t.replace('▁', '').strip()
    if sc and tc and len(sc) >= 1 and len(tc) >= 1:
        print(f"    ('{sc}', '{tc}'),")
print("]")

total = len(t1_combined) + len(t2_combined) + len(t3_combined)
print(f"\n# Total unique: {total}")
