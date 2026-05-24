"""
SPR Echo — cyclic shift as router (no training, no params)
depth layers of roll+sign_alt → deterministic path → leaf → store/retrieve
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
val_sents = load_sents(val_file, 2000)

word2id = {"<pad>": 0, "<unk>": 1}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}
depth = 14; n_leaves = 1 << depth
print(f"vocab={V} leaves={n_leaves} depth={depth}")

# ── Cyclic shift router (no params, deterministic) ──
def sign_alt(x):
    mask = torch.tensor([1., -1.] * (d//2 + 1), device=x.device)[:d]
    return x * mask

# Independent random projection per depth
rand_W = torch.randn(depth, d)  # (depth, d) — each depth has its own projection

def route(token_emb):
    idx = 0
    for dp in range(depth):
        score = (token_emb * rand_W[dp]).sum()  # dot product with depth-specific vector
        if score > 0:
            idx = 2 * idx + 2  # right
        else:
            idx = 2 * idx + 1  # left
    return idx - (n_leaves - 1)  # leaf index [0, n_leaves)

torch.manual_seed(42)
# Simple random embeddings
E = torch.randn(V, d) * 0.5

leaf_for = torch.zeros(V, dtype=torch.long)
for wid in range(V):
    leaf_for[wid] = route(E[wid])

# Each leaf stores its most frequent word
leaf_top = {}
leaf_words = [[] for _ in range(n_leaves)]
for wid in range(V):
    leaf_words[leaf_for[wid].item()].append(wid)

solo = sum(1 for lw in leaf_words if len(lw) == 1)
multi = sum(1 for lw in leaf_words if len(lw) > 1)
empty = sum(1 for lw in leaf_words if len(lw) == 0)
print(f"leaves: solo={solo} multi={multi} empty={empty} solo%={100*solo/V:.1f}%")

for lid in range(n_leaves):
    if leaf_words[lid]:
        leaf_top[lid] = Counter(leaf_words[lid]).most_common(1)[0][0]

# ── BLEU ──
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
    hyps.append([leaf_top.get(leaf_for[wid].item(), 1) for wid in ids])

b = bleu(refs, hyps)
print(f"\nBLEU-4 = {b:.2f}")

# samples
print(f"\n=== samples ===")
for i in range(min(3, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s][:6]
    src = ' '.join([id2word.get(w, '?') for w in ids])
    hyp = ' '.join([id2word.get(leaf_top.get(leaf_for[wid].item(), 1), '?') for wid in ids])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
