"""
SPR Echo — Residual Tree Cascade (Direction A)
Tree1 (depth=7, 128 leaves): coarse semantics
Tree2 (depth=7, 128 leaves): residuals E - leaf_center
Combined: 128×128 = 16384 effective leaves from 254 nodes
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
print(f"vocab={V} d={d}")

torch.manual_seed(42)
E = torch.randn(V, d).cuda()
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1)).cuda()[:d]

def spr_route_offset(embeddings, depth_start, depth_end):
    """Route using cyclic shifts from depth_start to depth_end-1"""
    V = len(embeddings); n_depths = depth_end - depth_start
    idx = torch.zeros(V, dtype=torch.long, device='cuda')
    current = embeddings.clone()
    for dp in range(depth_start, depth_end):
        current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK
        scores = (embeddings * current).sum(dim=-1)
        go_right = scores > 0
        idx = idx * 2 + 1
        idx[go_right] += 1
    return idx - ((1 << n_depths) - 1)

# ── Tree 1: coarse (shifts 0-6 → 7 depths → 128 leaves) ──
d1 = 7; l1 = 1 << d1
print(f"Tree1: shifts 0-6, depth={d1} leaves={l1}")
leaf1 = spr_route_offset(E, 0, 7)

# ── Tree 2: fine (shifts 7-13 → 7 depths → 128 leaves) ──
d2 = 7; l2 = 1 << d2
print(f"Tree2: shifts 7-13, depth={d2} leaves={l2}")
leaf2 = spr_route_offset(E, 7, 14)

# ── Combined leaf: leaf1 * l2 + leaf2 = 128*128 = 16384 ──
leaf_combined = leaf1 * l2 + leaf2
print(f"Effective leaves: {l1 * l2}")

# Stats
leaf_words = [[] for _ in range(l1 * l2)]
for wid in range(V):
    leaf_words[leaf_combined[wid].item()].append(wid)
solo = sum(1 for lw in leaf_words if len(lw) == 1)
multi = sum(1 for lw in leaf_words if len(lw) > 1)
empty = sum(1 for lw in leaf_words if len(lw) == 0)
print(f"combined: solo={solo} multi={multi} empty={empty} solo%={100*solo/V:.1f}%")

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

leaf_top = {}
for lid in range(l1 * l2):
    if leaf_words[lid]: leaf_top[lid] = leaf_words[lid][0]

refs, hyps = [], []
for s in val_sents[:500]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 4: continue
    refs.append(ids)
    hyps.append([leaf_top.get(leaf_combined[wid].item(), 1) for wid in ids])

b = bleu(refs, hyps)
print(f"\nResidual Cascade BLEU-4 = {b:.2f}")

# Compare: single tree depth=14
leaf_single = spr_route_offset(E, 0, 14)
lw_single = [[] for _ in range(1<<14)]
for wid in range(V): lw_single[leaf_single[wid].item()].append(wid)
solo_single = sum(1 for lw in lw_single if len(lw) == 1)
lt_single = {}
for lid in range(1<<14):
    if lw_single[lid]: lt_single[lid] = lw_single[lid][0]
hyp_single = []
for s in val_sents[:500]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 4: continue
    hyp_single.append([lt_single.get(leaf_single[wid].item(), 1) for wid in ids])
b_single = bleu(refs, hyp_single if hyp_single else [[1]])
solo_s = sum(1 for l in range(1<<14) if (leaf_single==l).sum().item()==1)

print(f"Single depth=14: solo={solo_s}/{V} ({100*solo_s/V:.1f}%) BLEU={b_single:.2f}")
print(f"Residual cascade: solo={solo}/{V} ({100*solo/V:.1f}%) BLEU={b:.2f}")
print(f"Delta: +{b - b_single:.2f}")
