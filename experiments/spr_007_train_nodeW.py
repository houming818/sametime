"""
SPR-007 v3 — char-ngram embeddings, frozen E, train node_W only
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os
from collections import Counter

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 64:
            raw.append(l.split("\t", 1)[1].strip().lower().split())

PAD = 8
word2id = {}; id2word = {}
for s in raw:
    for w in s[:PAD]:
        if w not in word2id: word2id[w] = len(word2id); id2word[len(id2word)] = w

V, d = len(word2id), 16
depth = 5
n_nodes, n_leaves = (1<<depth)-1, 1<<depth

# char n-gram embed
def char_emb(word, d):  # FIX: name it differently from variable E
    v = np.zeros(d)
    w = '#' + word + '#'
    for i in range(len(w)-2):
        v[abs(hash(w[i:i+3])) % d] += 1.0
    n = np.linalg.norm(v)
    return torch.tensor(v / n if n > 0 else v, dtype=torch.float32)

E_emb = torch.zeros(V, d)
for w in word2id: E_emb[word2id[w]] = char_emb(w, d)

print(f"vocab={V} leaves={n_leaves}")

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
    sc=[]; e=E_emb.numpy()
    for l in range(n_leaves):
        ids=(lm==l).nonzero(as_tuple=True)[0].numpy()
        if len(ids)>1:
            ee=e[ids]; ee=ee/(np.linalg.norm(ee,axis=1,keepdims=True)+1e-8)
            sim=np.dot(ee, ee.T); np.fill_diagonal(sim,0)
            sc.append(sim.mean())
    return float(np.mean(sc)) if sc else 0.0

# baseline
with torch.no_grad():
    a0 = soft_assign(E_emb, torch.randn(n_nodes,d), depth)
    l0 = a0.argmax(dim=1)
    g0 = leaf_groups(l0)
print(f"random: coh={coh(l0):.4f}  active={sum(1 for g in g0 if g)}")

# train
node_W = nn.Parameter(torch.randn(n_nodes,d)*0.05)
opt = torch.optim.Adam([node_W], lr=0.03)
for step in range(500):
    opt.zero_grad()
    assign = soft_assign(E_emb, node_W, depth)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E_emb / ls.unsqueeze(1)
    loss = F.mse_loss(E_emb, lc[assign.argmax(dim=1)])
    ideal = V/n_leaves
    loss = loss + 5.0*((ls-ideal)**2).mean()/ideal  # stronger balance
    loss.backward(); opt.step()

with torch.no_grad():
    a1 = soft_assign(E_emb, node_W.detach(), depth)
    l1 = a1.argmax(dim=1)
    g1 = leaf_groups(l1)
    active = sum(1 for g in g1 if g)

print(f"\ntrained: coh={coh(l1):.4f}  active={active}/{n_leaves}")
print(f"\n=== leaf content ===")
for lid, words in sorted(enumerate(g1), key=lambda x:-len(x[1]))[:5]:
    if len(words)<2: continue
    ws = [id2word[w] for w in words[:6]]
    # within-leaf sim
    ee = E_emb[words]; norm = ee.norm(dim=1).unsqueeze(1) + 1e-8; ee_chk = ee / norm
    s = (ee_chk @ ee_chk.T).numpy(); np.fill_diagonal(s,0)
    print(f"  leaf {lid} ({len(words)}w, sim={s.mean():.3f}): {ws}")
