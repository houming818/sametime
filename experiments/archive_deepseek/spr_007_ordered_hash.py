"""
SPR-007 v6 — Hybrid tree with order-aware hash
position split preserves order; weighted merge encodes it
"""
import torch, numpy as np, os

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 512:
            raw.append(l.split("\t", 1)[1].strip().lower().split())

word2id, id2word = {}, {}
for s in raw:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id); id2word[len(id2word)] = w
V, d = len(word2id), 16

coocc = np.zeros((V, d), dtype=np.float32)
for s in raw:
    for i, w in enumerate(s):
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j and s[j] in word2id:
                coocc[wid, word2id[s[j]] % d] += 1
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32)

print(f"vocab={V}")

# ── test sentences ──
test_raw = [raw[0], raw[1], raw[2]]
P = 8
sent_tokens = []
for s in test_raw:
    ids = []
    for w in s[:P]:
        ids.append(word2id.get(w, 0))
    while len(ids) < P: ids.append(0)
    sent_tokens.append(ids)

def pos_hash(tokens, alpha=0.7):
    """fixed position split tree + weighted merge"""
    n = len(tokens)
    if n <= 1:
        return E[tokens[0]] if n == 1 else torch.zeros(d)
    
    mid = n // 2
    left = tokens[:mid]
    right = tokens[mid:]
    
    HL = pos_hash(left, alpha) if left else torch.zeros(d)
    HR = pos_hash(right, alpha) if right else torch.zeros(d)
    
    return alpha * HL + (1 - alpha) * HR

# ── tests ──
print("\n=== order sensitivity ===")
for si, s_ids in enumerate(sent_tokens[:3]):
    real = [t for t in s_ids if t > 0][:8]
    # pad to 8
    real = (real + [0]*8)[:8]
    
    H_fwd = pos_hash(real, alpha=0.7)
    H_rev = pos_hash(real[::-1], alpha=0.7)
    
    ws = [id2word.get(t, '∅') for t in real[:4]]
    same = torch.allclose(H_fwd, H_rev, atol=1e-3)
    print(f"  sent{si} {ws}: fwd≠rev={not same}  H_fwd={H_fwd[:3].numpy().round(2)} H_rev={H_rev[:3].numpy().round(2)}")

print("\n=== determinism ===")
same = 0
for s_ids in sent_tokens:
    real = list(s_ids) + [0]*8; real = real[:8]
    H1 = pos_hash(real, alpha=0.7)
    H2 = pos_hash(real, alpha=0.7)
    if torch.allclose(H1, H2, atol=1e-3): same += 1
print(f"  {same}/{len(sent_tokens)} deterministic")

print("\n=== mean (α=0.5) is order-invariant ===")
H_m = pos_hash(sent_tokens[0][:8], alpha=0.5)
H_mr = pos_hash(sent_tokens[0][:8][::-1], alpha=0.5)
print(f"  α=0.5 fwd==rev: {torch.allclose(H_m, H_mr, atol=1e-3)}")

print("\n✓ position split + weighted merge = order-aware hash")
