"""
SPR Echo Pure — Deterministic self-routing via cyclic shift evolution
Each depth: roll(e, dp) * SIGN_MASK → dot with original → path decision
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

print("loading data...")
train_sents = load_sents(train_file, 5000)
val_sents = load_sents(val_file, 200)

word2id = {"<pad>": 0, "<unk>": 1}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}

depth = 14; n_leaves = 1 << depth
print(f"vocab={V} d={d} depth={depth} leaves={n_leaves}")

torch.manual_seed(42)
E = torch.randn(V, d).cuda()
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)

SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1)).cuda()[:d]

def spr_self_route(token_emb):
    """SPR self-routing: token evolves through cyclic shifts.
       Current vector = roll(prev, dp+1) * SIGN_MASK
       Routing = dot(original, current) > 0"""
    idx = 0
    current = token_emb.clone()
    for dp in range(depth):
        current = torch.roll(current, shifts=dp + 1, dims=-1) * SIGN_MASK
        score = torch.dot(token_emb, current)
        if score > 0:
            idx = 2 * idx + 2  # right
        else:
            idx = 2 * idx + 1  # left
    return idx - (n_leaves - 1)

print("\nrunning SPR self-routing...")
leaf_for = torch.zeros(V, dtype=torch.long)
for wid in range(V):
    leaf_for[wid] = spr_self_route(E[wid])

leaf_words = [[] for _ in range(n_leaves)]
for wid in range(V):
    leaf_words[leaf_for[wid].item()].append(wid)

solo = sum(1 for lw in leaf_words if len(lw) == 1)
multi = sum(1 for lw in leaf_words if len(lw) > 1)
empty = sum(1 for lw in leaf_words if len(lw) == 0)
print(f"SPR self-routing: solo={solo} multi={multi} empty={empty} solo%={100*solo/V:.1f}%")
print(f"avg words/leaf (active): {V/(solo+multi):.1f}")

leaf_top = {}
for lid in range(n_leaves):
    if leaf_words[lid]:
        leaf_top[lid] = leaf_words[lid][0]

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
    hyps.append([leaf_top.get(leaf_for[wid].item(), 1) for wid in ids])

b = bleu(refs, hyps)
print(f"\nSPR Deterministic Echo BLEU-4 = {b:.2f}")

# Determinism
leaf_for2 = torch.zeros(V, dtype=torch.long)
for wid in range(V):
    leaf_for2[wid] = spr_self_route(E[wid])
same = (leaf_for == leaf_for2).sum().item()
print(f"deterministic: {same}/{V}")

# Samples
print(f"\n=== samples ===")
for i in range(min(3, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s][:6]
    src = ' '.join([id2word.get(w, '?') for w in ids])
    hyp = ' '.join([id2word.get(leaf_top.get(leaf_for[wid].item(), 1), '?') for wid in ids])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
