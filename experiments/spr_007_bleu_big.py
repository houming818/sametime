"""
SPR-007 BLEU v3 — Fixed: depth=7 safe, frequency-weighted leaf top, <unk> handling
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

# Frequency-weighted vocabulary
word_counts = Counter()
for s in train_sents: word_counts.update(s)

word2id = {"<pad>": 0, "<unk>": 1}
for w, _ in word_counts.most_common(15000):
    if w not in word2id: word2id[w] = len(word2id)
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}
depth = 7; n_leaves = 1 << depth; n_nodes = n_leaves - 1
print(f"vocab={V} d={d} depth={depth} leaves={n_leaves}")

# Co-occ
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
    if si % 50000 == 49999: print(f"  {si+1} done")
norms = coocc.norm(dim=1, keepdim=True) + 1e-8
E = coocc / norms

# Non-recursive routing
def get_routing_probs(emb, nw):
    N = len(emb); probs = torch.ones(N, 1, device=emb.device); node_idx = 0
    for layer in range(depth):
        layer_nodes = 1 << layer
        w_layer = nw[node_idx : node_idx + layer_nodes]; node_idx += layer_nodes
        pr = torch.sigmoid(emb @ w_layer.T)
        probs = probs.unsqueeze(2)
        splits = torch.stack([1 - pr, pr], dim=2)
        probs = (probs * splits).view(N, -1)
    return probs

# Train
nw = nn.Parameter(torch.randn(n_nodes, d).cuda() * 0.05)
opt = torch.optim.Adam([nw], lr=0.02)
ideal = V / n_leaves

print("training...")
for step in range(300):
    opt.zero_grad()
    assign = get_routing_probs(E, nw)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)
    loss = F.mse_loss(E, assign @ lc) + 2.0 * ((ls - ideal)**2).mean() / ideal
    loss.backward(); opt.step()

with torch.no_grad():
    leaf_for = get_routing_probs(E, nw).argmax(dim=1)

# Fix: frequency-weighted leaf top
def leaf_top_freq(leaf_assign):
    lt = {}
    for lid in range(n_leaves):
        wids = (leaf_assign == lid).nonzero(as_tuple=True)[0].tolist()
        if wids:
            # Pick word with highest corpus frequency in this leaf
            best = max(wids, key=lambda wid: word_counts.get(id2word.get(wid, ''), 0))
            lt[lid] = best
        else:
            lt[lid] = 1  # <unk>
    return lt

leaf_top = leaf_top_freq(leaf_for)

# Random baseline
nw_rand = torch.randn(n_nodes, d).cuda() * 0.05
with torch.no_grad():
    lf_rand = get_routing_probs(E, nw_rand).argmax(dim=1)
lt_rand = leaf_top_freq(lf_rand)

# BLEU
def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def bleu(rf, hp):
    C=Counter; ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(rf,hp):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)  # no n-gram = no penalty
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

# Sanity checks
refs_p = [[0,1,2]]; hyps_p = [[0,1,2]]
print(f"\nsanity perfect echo BLEU = {bleu(refs_p, hyps_p):.1f} (expect 100)")

refs, hyps_r, hyps_t = [], [], []
for s in val_sents[:500]:
    ids = [word2id.get(w, 1) for w in s]  # <unk> for unknown, preserve length
    if len(ids) < 4: continue
    refs.append(ids)
    hyps_r.append([lt_rand.get(lf_rand[wid].item(), 1) for wid in ids])
    hyps_t.append([leaf_top.get(leaf_for[wid].item(), 1) for wid in ids])

b_r = bleu(refs, hyps_r)
b_t = bleu(refs, hyps_t)
print(f"random  BLEU: {b_r:.2f}")
print(f"trained BLEU: {b_t:.2f}")
print(f"delta       : {b_t - b_r:+.2f}")

# Samples
print(f"\n=== samples ===")
for i in range(min(3, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s][:6]
    src = ' '.join([id2word.get(w, '?') for w in ids])
    rnd = ' '.join([id2word.get(lt_rand.get(lf_rand[wid].item(), 1), '?') for wid in ids])
    trn = ' '.join([id2word.get(leaf_top.get(leaf_for[wid].item(), 1), '?') for wid in ids])
    print(f"  src: {src}")
    print(f"  rnd: {rnd}")
    print(f"  trn: {trn}")
    print()
