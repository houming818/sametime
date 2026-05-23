"""
SPR-007 v4 — WMT14 self-trained co-occurrence embeddings
no external download — learn from the data itself
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os
from collections import Counter

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 512:  # more sentences for better co-occur
            raw.append(l.split("\t", 1)[1].strip().lower().split())

# build vocab from all sentences
word2id, id2word = {}, {}
for s in raw:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id); id2word[len(id2word)] = w

V, d = len(word2id), 16

# co-occurrence embedding: count how often words appear together (window=3)
coocc = np.zeros((V, d), dtype=np.float32)
for s in raw:
    for i, w in enumerate(s):
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j:
                cw = s[j]
                if cw in word2id:
                    cwid = word2id[cw]
                    coocc[wid, cwid % d] += 1

# normalize
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
coocc = coocc / norms
E_coocc = torch.tensor(coocc, dtype=torch.float32)

print(f"vocab={V} d={d} sentences={len(raw)}")

# show similarity
def sim(w1, w2):
    if w1 in word2id and w2 in word2id:
        return float(F.cosine_similarity(E_coocc[word2id[w1]].unsqueeze(0), E_coocc[word2id[w2]].unsqueeze(0)).item())
    return 0

print(f"\nco-occ embedding sims:")
for a,b in [('president','obama'),('strategy','plan'),('the','cat'),('court','law')]:
    if a in word2id and b in word2id:
        print(f"  '{a}' ⇔ '{b}': {sim(a,b):.3f}")
    else:
        print(f"  '{a}' ⇔ '{b}': N/A")

# ── tree routing ──
depth = 8  # FORCED: 256 leaves for 3085 words → avg 12/leaf
n_leaves = 1<<depth; n_nodes = n_leaves-1
print(f"\ntree: depth={depth} leaves={n_leaves}")

def soft_assign(emb, nw, depth, idx=0):
    if depth==0: return torch.ones(len(emb),1)
    sc = emb @ nw[idx]; pr = torch.sigmoid(sc)
    ch = soft_assign(emb, nw, depth-1, 2*idx+1)
    L = ch.shape[1]; a = torch.zeros(len(emb), L*2)
    a[:,:L] = (1-pr).unsqueeze(1)*ch; a[:,L:] = pr.unsqueeze(1)*ch
    return a

def leaf_groups(lm):
    g = [[] for _ in range(n_leaves)]
    for w in range(V): g[lm[w].item()].append(w)
    return g

def coh(lm):
    sc=[]; e=E_coocc.numpy()
    for l in range(n_leaves):
        ids=(lm==l).nonzero(as_tuple=True)[0].numpy()
        if len(ids)>1:
            ee=e[ids]; ee=ee/(np.linalg.norm(ee,axis=1,keepdims=True)+1e-8)
            s=ee @ ee.T; np.fill_diagonal(s,0)
            sc.append(s.mean())
    return float(np.mean(sc)) if sc else 0.0

# random baseline
with torch.no_grad():
    a0 = soft_assign(E_coocc, torch.randn(n_nodes,d), depth)
    l0 = a0.argmax(dim=1)
print(f"random: coh={coh(l0):.4f}  active={sum(1 for g in leaf_groups(l0) if g)}")

# train node_W only
node_W = nn.Parameter(torch.randn(n_nodes,d)*0.05)
opt = torch.optim.Adam([node_W], lr=0.03)
for step in range(500):
    opt.zero_grad()
    assign = soft_assign(E_coocc, node_W, depth)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E_coocc / ls.unsqueeze(1)
    loss = F.mse_loss(E_coocc, lc[assign.argmax(dim=1)])
    ideal = V/n_leaves
    loss = loss + 5.0*((ls-ideal)**2).mean()/ideal
    loss.backward(); opt.step()

with torch.no_grad():
    a1 = soft_assign(E_coocc, node_W.detach(), depth)
    l1 = a1.argmax(dim=1)
    g1 = leaf_groups(l1)
    active = sum(1 for g in g1 if g)

print(f"\ntrained: coh={coh(l1):.4f}  active={active}/{n_leaves} (Δ={coh(l1)-coh(l0):+.3f})")
print(f"\n=== leaf content ===")
for lid, words in sorted(enumerate(g1), key=lambda x:-len(x[1]))[:6]:
    if len(words)<2: continue
    ws = [id2word[w] for w in words[:6]]
    ee = E_coocc[words]; n = ee.norm(dim=1).unsqueeze(1) + 1e-8
    s = (ee/n @ (ee/n).T).numpy(); np.fill_diagonal(s,0)
    print(f"  leaf {lid} ({len(words)}w, sim={s.mean():.3f}): {ws}")
