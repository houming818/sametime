"""
SPR-007 experiment — ordered hash on real WMT14 sentences
Verify: determinism, order sensitivity, semantic proximity
"""
import torch, numpy as np, os
from collections import Counter

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 512:
            raw.append(l.split("\t", 1)[1].strip().lower().split())

# co-occ embeddings
word2id = {}
for s in raw: 
    for w in s: 
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 32

coocc = np.zeros((V, d), dtype=np.float32)
for s in raw[:256]:
    for i, w in enumerate(s):
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j and s[j] in word2id:
                coocc[wid, word2id[s[j]] % d] += 1
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32)

def sign_alt(x):
    mask = torch.tensor([1., -1.] * (x.shape[-1] // 2 + 1))[:x.shape[-1]]
    return x * mask

def ordered_hash(tokens, depth=0):
    """Cyclic shift + sign alternation"""
    if len(tokens) <= 1:
        return E[tokens[0]] if len(tokens)==1 else torch.zeros(d)
    mid = len(tokens)//2
    HL = ordered_hash(tokens[:mid], depth+1)
    HR = ordered_hash(tokens[mid:], depth+1)
    return HL + sign_alt(torch.roll(HR, shifts=depth+1, dims=-1))

def unordered_hash(tokens):
    """Mean hash (baseline)"""
    return E[tokens].mean(dim=0)

# ── test sentences ──
test_sents = []
for s in raw[256:320]:
    ids = [word2id[w] for w in s if w in word2id][:16]
    if len(ids) >= 4:
        # pad to power of 2
        n = 1; 
        while n < len(ids): n *= 2
        ids.extend([0] * (n - len(ids)))
        test_sents.append(ids)
    if len(test_sents) >= 20: break

print(f"vocab={V} test_sents={len(test_sents)}")

# ── 1. determinism ──
same_ord = same_unord = 0
for ids in test_sents:
    H1 = ordered_hash(ids)
    H2 = ordered_hash(ids)
    H1u = unordered_hash(ids)
    H2u = unordered_hash(ids)
    if torch.allclose(H1, H2, atol=1e-3): same_ord += 1
    if torch.allclose(H1u, H2u, atol=1e-3): same_unord += 1

# ── 2. order sensitivity ──
order_sensitive = 0
for ids in test_sents:
    H_fwd = ordered_hash(ids)
    H_rev = ordered_hash(ids[::-1])
    if not torch.allclose(H_fwd, H_rev, atol=1e-3):
        order_sensitive += 1

# ── 3. sentence discrimination ──
# Each sentence's hash should be unique
hashes_ord = torch.stack([ordered_hash(ids) for ids in test_sents])
hashes_unord = torch.stack([unordered_hash(ids) for ids in test_sents])

# pairwise cosine distances
def pairwise_unique(hashes):
    sim = hashes @ hashes.T / (hashes.norm(dim=1).unsqueeze(1) * hashes.norm(dim=1).unsqueeze(0) + 1e-8)
    mask = torch.eye(len(hashes)) == 0
    return sim[mask].mean().item()

unique_ord = pairwise_unique(hashes_ord)
unique_unord = pairwise_unique(hashes_unord)

print(f"\n=== Results on {len(test_sents)} WMT14 sentences ===")
print(f"determinism:     ordered={same_ord}/{len(test_sents)}  unordered={same_unord}/{len(test_sents)}")
print(f"order sensitive: {order_sensitive}/{len(test_sents)}")
print(f"avg inter-sent sim: ordered={unique_ord:.3f}  unordered={unique_unord:.3f}")
print(f"ordered hash MORE discriminative: {'yes' if unique_ord < unique_unord else 'no'}")

# ── 4. sample: show hash values for different sentences ──
print(f"\n=== hash samples ===")
id2word = {v:k for k,v in word2id.items()}
for i in [0, 1, 2]:
    ids = test_sents[i]
    words = [id2word.get(w, '∅') for w in ids[:4]]
    H_ord = ordered_hash(ids)
    H_unord = unordered_hash(ids)
    print(f"  sent {i}: {words}")
    print(f"    ordered: {H_ord[:4].numpy().round(2)}")
    print(f"    unordered: {H_unord[:4].numpy().round(2)}")
