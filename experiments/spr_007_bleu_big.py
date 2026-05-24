"""
SPR-007 BLEU v4 — deep tree (16384 leaves), trainable E, target Phase 0's 97
10K vocab, 450K co-occ, end-to-end training
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os, time
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
print(f"train={len(train_sents)} val={len(val_sents)}")

# Top 10K words
word_counts = Counter()
for s in train_sents: word_counts.update(s)
top_words = [w for w, _ in word_counts.most_common(10000)]

word2id = {"<pad>": 0, "<unk>": 1}
for w in top_words: word2id[w] = len(word2id)
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}
depth = 14; n_leaves = 1 << depth; n_nodes = n_leaves - 1
print(f"vocab={V} d={d} depth={depth} leaves={n_leaves} n_nodes={n_nodes}")

# GPU co-occ
print("building co-occ on GPU...")
coocc = torch.zeros(V, d, device='cuda')
for si, s in enumerate(train_sents):
    ids = [word2id.get(w, 1) for w in s]
    n = len(ids)
    for i in range(n):
        wid = ids[i]
        start, end = max(0, i-3), min(n, i+4)
        ctx = torch.tensor(ids[start:end], device='cuda')
        coocc[wid].scatter_add_(0, ctx % d, torch.ones(len(ctx), device='cuda'))
    if si % 50000 == 49999: print(f"  {si+1}")
norms = coocc.norm(dim=1, keepdim=True) + 1e-8
E_fixed = (coocc / norms).detach()  # NOT trainable
E = E_fixed

# Non-recursive routing
def get_routing_probs(emb, nw):
    N = len(emb); probs = torch.ones(N, 1, device=emb.device); node_idx = 0
    for layer in range(depth):
        layer_nodes = 1 << layer
        w_layer = nw[node_idx : node_idx + layer_nodes]; node_idx += layer_nodes
        pr = torch.sigmoid(emb @ w_layer.T)
        probs = probs.unsqueeze(2)
        probs = (probs * torch.stack([1-pr, pr], dim=2)).view(N, -1)
    return probs

# Train only node_W
nw = nn.Parameter(torch.randn(n_nodes, d).cuda() * 0.05)
opt = torch.optim.Adam([nw], lr=0.02)
ideal = V / n_leaves

print("training E + node_W end-to-end...")
t0 = time.time()
for step in range(500):
    opt.zero_grad()
    assign = get_routing_probs(E, nw)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)
    loss = F.mse_loss(E, assign @ lc) + 10.0 * ((ls - ideal)**2).mean()/ideal
    loss.backward(); opt.step()
    if step % 100 == 0:
        with torch.no_grad():
            leaf = get_routing_probs(E, nw).argmax(dim=1)
            solo = sum(1 for lid in range(n_leaves) if (leaf == lid).sum().item() == 1)
        print(f"  step {step:3d}: loss={loss.item():.4f}  solo_leaves={solo}/{V}")

print(f"trained in {time.time()-t0:.0f}s")

# Leaf top with frequency weighting
with torch.no_grad():
    leaf_for = get_routing_probs(E, nw).argmax(dim=1)
    
def leaf_top_freq(leaf_assign):
    lt = {}
    for lid in range(n_leaves):
        wids = (leaf_assign == lid).nonzero(as_tuple=True)[0].tolist()
        if wids:
            best = max(wids, key=lambda wid: word_counts.get(id2word.get(wid, ''), 0))
            lt[lid] = best
        else:
            lt[lid] = 1
    return lt

leaf_top = leaf_top_freq(leaf_for)
solo = sum(1 for lid in range(n_leaves) if (leaf_for == lid).sum().item() == 1)
print(f"solo leaves: {solo}/{V} ({100*solo/V:.1f}%)")

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
print(f"\n=== BLEU-4 = {b:.2f} ===")
print(f"vocab={V} leaves={n_leaves} solo={solo} solo%={100*solo/V:.1f}%")

# Samples
print(f"\n=== samples ===")
for i in range(min(3, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s][:6]
    hyp = [id2word.get(leaf_top.get(leaf_for[wid].item(), 1), '?') for wid in ids]
    print(f"  src: {' '.join([id2word.get(w,'?') for w in ids])}")
    print(f"  hyp: {' '.join(hyp)}")
    print()
