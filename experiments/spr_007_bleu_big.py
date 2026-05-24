"""
SPR-007 BLEU — 50K WMT14, train routing, measure BLEU on reconstruction
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

word2id = {"<pad>": 0}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}
depth = 8; n_leaves = 1 << depth; n_nodes = n_leaves - 1
print(f"vocab={V} d={d} depth={depth} leaves={n_leaves}")

# co-occ
print("building co-occ...")
coocc = np.zeros((V, d), dtype=np.float32)
for s in train_sents[:10000]:
    for i, w in enumerate(s):
        if w not in word2id: continue
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j and s[j] in word2id: coocc[wid, word2id[s[j]] % d] += 1
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32).cuda()

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

# ── Train node_W ──
nw = nn.Parameter(torch.randn(n_nodes, d).cuda() * 0.05)
opt = torch.optim.Adam([nw], lr=0.01)
ideal = V / n_leaves

print("training routing...")
for step in range(300):
    opt.zero_grad()
    assign = get_routing_probs(E, nw)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)
    loss = F.mse_loss(E, assign @ lc) + 2.0 * ((ls - ideal)**2).mean()/ideal
    loss.backward(); opt.step()

with torch.no_grad():
    leaf_for = get_routing_probs(E, nw).argmax(dim=1)

# ── BLEU: word → leaf → top word at leaf ──
leaf_top = {}
for lid in range(n_leaves):
    ids = (leaf_for == lid).nonzero(as_tuple=True)[0]
    if len(ids) > 0:
        leaf_top[lid] = Counter(ids.tolist()).most_common(1)[0][0]

def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def bleu(rf, hp):
    C=Counter; ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(rf,hp):
            rc=C(ng(r,n)); hc=C(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1))
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

# Random baseline
nw_rand = torch.randn(n_nodes, d).cuda() * 0.05
with torch.no_grad():
    lf_rand = get_routing_probs(E, nw_rand).argmax(dim=1)
    lt_rand = {}
    for lid in range(n_leaves):
        ids = (lf_rand == lid).nonzero(as_tuple=True)[0]
        if len(ids) > 0: lt_rand[lid] = Counter(ids.tolist()).most_common(1)[0][0]

refs, hyps_rand, hyps_trained = [], [], []
for s in val_sents[:500]:
    ids = [word2id.get(w, 0) for w in s if word2id.get(w, 0) > 0]
    if len(ids) < 3: continue
    refs.append(ids)
    hyps_rand.append([lt_rand.get(lf_rand[wid].item(), 0) for wid in ids])
    hyps_trained.append([leaf_top.get(leaf_for[wid].item(), 0) for wid in ids])

b_rand = bleu(refs, hyps_rand)
b_trained = bleu(refs, hyps_trained)

print(f"\n=== BLEU-4 ===")
print(f"random  : {b_rand:.2f}")
print(f"trained : {b_trained:.2f}")
print(f"delta   : {b_trained - b_rand:+.2f}")
print(f"improved: {'YES' if b_trained > b_rand else 'NO'}")

# samples
print(f"\n=== samples ===")
for i in [0, 1, 2]:
    s = val_sents[i]
    ids = [word2id.get(w, 0) for w in s if word2id.get(w, 0) > 0][:6]
    hyp_r = [id2word.get(lt_rand.get(lf_rand[wid].item(), 0), '?') for wid in ids]
    hyp_t = [id2word.get(leaf_top.get(leaf_for[wid].item(), 0), '?') for wid in ids]
    print(f"  src: {' '.join([id2word[w] for w in ids[:5]])}")
    print(f"  rnd: {' '.join(hyp_r[:5])}")
    print(f"  trn: {' '.join(hyp_t[:5])}")
    print()
