"""
Extract anchor candidates from parallel corpus via co-occurrence statistics.
Scans EN-ZH parallel sentences, counts BPE token co-occurrences,
filters high-confidence pairs, outputs candidate list with examples.
"""
import sentencepiece as spm, sys
from collections import defaultdict, Counter

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
print(f"BPE vocab={V}")

# Load parallel sentences
print("loading parallel corpus...")
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh, en = l.strip().split("\t", 1)
            zh_t = sp.encode_as_ids(zh.strip())
            en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 2 and len(en_t) >= 2:
                pairs.append((zh_t, en_t))
print(f"  {len(pairs)} parallel pairs")

# Sample sentences for examples (store first 3 per EN-ZH pair)
examples = defaultdict(list)

# Count co-occurrences: for each EN token, count ZH tokens
en_to_zh = defaultdict(Counter)
n_sents = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
n_sents = min(n_sents, len(pairs))

print(f"scanning {n_sents} sentences...")
# Only consider tokens with enough occurrences
en_freq = Counter()
zh_freq = Counter()
for zh, en in pairs[:n_sents]:
    for e in set(en): en_freq[e] += 1
    for z in set(zh): zh_freq[z] += 1

min_tok_freq = int(sys.argv[3]) if len(sys.argv) > 3 else 10
valid_en = {e for e, c in en_freq.items() if c >= min_tok_freq}
valid_zh = {z for z, c in zh_freq.items() if c >= min_tok_freq}
print(f"  EN tokens with freq>={min_tok_freq}: {len(valid_en)}")
print(f"  ZH tokens with freq>={min_tok_freq}: {len(valid_zh)}")

for idx, (zh, en) in enumerate(pairs[:n_sents]):
    if idx % 20000 == 0 and idx > 0:
        print(f"  {idx}/{n_sents}")
    en_set = {e for e in set(en) if e in valid_en}
    zh_set = {z for z in set(zh) if z in valid_zh}
    en_set = set(en)
    zh_set = set(zh)
    for e in en_set:
        for z in zh_set:
            en_to_zh[e][z] += 1
            key = (e, z)
            if len(examples[key]) < 3:
                examples[key].append((zh, en))

print("  done scanning")

# Extract top pairs per EN token
min_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print(f"\nextracting pairs with co-occurrence >= {min_count}...")

candidates = []
seen = set()
for e, zh_counts in en_to_zh.items():
    if not zh_counts: continue
    en_piece = sp.id_to_piece(e)
    # Skip pure punctuation/markers
    if en_piece in ('<unk>', '<s>', '</s>', '▁', '.', ',', '?', '!', ':', ';'):
        continue
    # Take top ZH token by co-occurrence
    top_zh, top_count = zh_counts.most_common(1)[0]
    if top_count < min_count: continue

    zh_piece = sp.id_to_piece(top_zh)
    # Skip if ZH is pure punctuation
    if zh_piece in ('<unk>', '<s>', '</s>', '▁', '.', ',', '?', '!', ':', ';'):
        continue

    # Compute PMI-like confidence
    # Confidence = co-occurrence / total_EN_occurrences
    total_e = sum(zh_counts.values())
    confidence = top_count / total_e

    key = (e, top_zh)
    if key not in seen:
        seen.add(key)
        candidates.append((e, top_zh, top_count, confidence, en_piece, zh_piece))

# Sort by confidence (best pairs first)
candidates.sort(key=lambda x: -x[2])  # sort by raw count

print(f"  total candidates: {len(candidates)}")

# Output tiers
high = [c for c in candidates if c[3] >= 0.3 and c[2] >= 10]
mid = [c for c in candidates if 0.1 <= c[3] < 0.3 and c[2] >= 5]
low = [c for c in candidates if c[3] < 0.1 and c[2] >= min_count]

print(f"  HIGH (conf>=0.3, count>=10): {len(high)}")
print(f"  MEDIUM (conf>=0.1, count>=5): {len(mid)}")
print(f"  LOW (conf<0.1, count>={min_count}): {len(low)}")

# Print HIGH candidates with examples
print(f"\n=== HIGH confidence candidates (top 50) ===")
for i, (e, z, count, conf, ep, zp) in enumerate(high[:50]):
    print(f"\n--- {i+1}: {ep:20s} → {zp:20s}  (count={count}, conf={conf:.2f}) ---")
    key = (e, z)
    for j, (zh_sent, en_sent) in enumerate(examples.get(key, [])[:3]):
        en_text = sp.decode_ids(en_sent)[:80]
        zh_text = sp.decode_ids(zh_sent)[:80]
        print(f"  ex{j+1}: EN: {en_text}")
        print(f"       ZH: {zh_text}")

# Save
with open("/tmp/anchor_candidates.txt", "w") as f:
    for e, z, count, conf, ep, zp in candidates:
        f.write(f"{e}\t{z}\t{count}\t{conf:.4f}\t{ep}\t{zp}\n")

# Also print all HIGH pairs as Python list for easy integration
print(f"\n=== All HIGH pairs ({len(high)}) — ready for ANCHOR_WORDS ===")
print("[")
for e, z, count, conf, ep, zp in high:
    print(f"    ('{ep}', '{zp}'),  # count={count} conf={conf:.2f}")
print("]")

print(f"\n=== All MEDIUM pairs ({len(mid)}) ===")
print("[")
for e, z, count, conf, ep, zp in mid[:100]:
    print(f"    ('{ep}', '{zp}'),  # count={count} conf={conf:.2f}")
if len(mid) > 100:
    print(f"    ... ({len(mid)-100} more)")
print("]")
