"""
SPR Echo — Decomposed feature routing
Split d-dim embedding into K chunks → each routes independently
Combined leaf = leaf_0 * L^(K-1) + leaf_1 * L^(K-2) + ... = L^K effective leaves
d=64, K=4, depth=7 → 128^4 = 268M leaves → near-zero collision
"""
import torch, numpy as np, math, os
from collections import Counter

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t", 1)[1].strip().lower().split())
    return sents

print("loading...")
train_sents = load_sents(train_file, 50000)
val_sents = load_sents(val_file, 500)

word2id = {"<pad>": 0, "<unk>": 1}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}

# Decompose: K=4 chunks, each routes depth=7 (128 leaves)
K = 4; chunk_d = d // K; chunk_depth = 7; chunk_leaves = 1 << chunk_depth
total_leaves = chunk_leaves ** K  # 128^4 = 268M
print(f"vocab={V} d={d} K={K} chunk_dim={chunk_d} chunk_leaves={chunk_leaves}")
print(f"total effective leaves = {chunk_leaves}^{K} = {total_leaves:,}")
print(f"words/leaf ratio = {V}/{total_leaves:.0f} = {V/total_leaves:.6f}")

torch.manual_seed(42)
E = torch.randn(V, d).cuda()
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
SIGN_MASK = torch.tensor([1., -1.] * (chunk_d//2 + 1)).cuda()[:chunk_d]

def route_chunk(chunks, depth):
    """Route a chunk (V, chunk_d) through depth levels of self-routing"""
    V = len(chunks)
    idx = torch.zeros(V, dtype=torch.long, device='cuda')
    current = chunks.clone()
    for dp in range(depth):
        current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK
        scores = (chunks * current).sum(dim=-1)
        go_right = scores > 0
        idx = idx * 2 + 1
        idx[go_right] += 1
    return idx - (chunk_leaves - 1)

# Route each chunk independently
leaf_chunks = torch.zeros(V, K, dtype=torch.long, device='cuda')
print("routing K chunks...")
for k in range(K):
    chunk_emb = E[:, k*chunk_d:(k+1)*chunk_d]
    leaf_chunks[:, k] = route_chunk(chunk_emb, chunk_depth)
    solo_k = (leaf_chunks[:, k].unique(return_counts=True)[1] == 1).sum().item()
    print(f"  chunk {k}: solo={solo_k}/{V} ({100*solo_k/V:.1f}%)")

# Combined: base-K encoding
leaf_combined = torch.zeros(V, dtype=torch.long).cuda()
for k in range(K-1, -1, -1):
    leaf_combined = leaf_combined * chunk_leaves + leaf_chunks[:, k]

leaf_words = [[] for _ in range(min(V, 1000000))]
max_leaf = leaf_combined.max().item()
# Use dict for sparse large-range leaves instead of list
from collections import defaultdict
leaf_words_dict = defaultdict(list)
for wid in range(V):
    leaf_words_dict[leaf_combined[wid].item()].append(wid)

solo = sum(1 for lw in leaf_words_dict.values() if len(lw) == 1)
multi = sum(1 for lw in leaf_words_dict.values() if len(lw) > 1)
print(f"\ncombined: solo={solo} multi={multi} active_leaves={len(leaf_words_dict)}")
print(f"solo% = {100*solo/V:.1f}%")

# Leaf top
leaf_top = {}
for lid, words in leaf_words_dict.items():
    leaf_top[lid] = Counter(words).most_common(1)[0][0]

# BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def bleu(rf, hp):
    C=Counter; ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(rf,hp):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

refs, hyps = [], []
for s in val_sents[:500]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 4: continue
    refs.append(ids)
    hyps.append([leaf_top[leaf_combined[wid].item()] for wid in ids])

b = bleu(refs, hyps)
print(f"\nBLEU-4 = {b:.2f}")

print(f"\n=== samples ===")
for i in range(min(3, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s][:6]
    src = ' '.join([id2word.get(w, '?') for w in ids])
    hyp = ' '.join([id2word.get(leaf_top[leaf_combined[wid].item()], '?') for wid in ids])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
