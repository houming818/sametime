"""
SPR-007: trained routing with strong balance constraint
key fix: balance_loss weight 10x stronger
decode = leaf's most frequent word
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
word2id = {}
sent_ids = []
for s in raw:
    ids = [0]*PAD
    for i, w in enumerate(s[:PAD]):
        if w not in word2id: word2id[w] = len(word2id)
        ids[i] = word2id[w]
    sent_ids.append(ids)

V, d = len(word2id), 16
depth = 5
n_nodes, n_leaves = (1<<depth)-1, 1<<depth
id2word = {v:k for k,v in word2id.items()}

print(f"vocab={V} leaves={n_leaves} params={n_nodes*d}")

torch.manual_seed(42)
E = nn.Parameter(torch.randn(V,d)*0.5)
node_W = nn.Parameter(torch.randn(n_nodes,d)*0.05)

def soft_assign(emb, nw, depth, idx=0):
    if depth==0: return torch.ones(len(emb),1)
    sc = emb @ nw[idx]; pr = torch.sigmoid(sc)
    ch = soft_assign(emb, nw, depth-1, 2*idx+1)
    L = ch.shape[1]; a = torch.zeros(len(emb), L*2)
    a[:,:L] = (1-pr).unsqueeze(1)*ch; a[:,L:] = pr.unsqueeze(1)*ch
    return a

def leaf_groups(leaf_map):
    g = [[] for _ in range(n_leaves)]
    for wid in range(V): g[leaf_map[wid].item()].append(wid)
    return g

def sent_bleu(leaf_map, top):
    refs, hyps = [], []
    for s_ids in sent_ids:
        ref = [t for t in s_ids if t > 0]
        hyp = []
        for tid in s_ids:
            if tid <= 0: continue
            l = leaf_map[tid].item()
            if l >= 0 and top[l] is not None:
                hyp.append(top[l])
        if hyp: refs.append(ref); hyps.append(hyp)
    if not refs: return 0.0
    def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    ps=[]
    for n in range(1,5):
        mch,ttl=0,0
        for r,h in zip(refs,hyps):
            rc=Counter(ng(r,n)); hc=Counter(ng(h,n))
            ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
        ps.append(mch/max(ttl,1))
    bpv=[1-len(r)/max(len(h),1) for r,h in zip(refs,hyps) if len(h)>0]
    bp=min(1.0,math.exp(max(bpv) if bpv else 0))
    return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

def coh(leaf_map, emb):
    sc=[]
    for l in range(n_leaves):
        ids=(leaf_map==l).nonzero(as_tuple=True)[0]
        if len(ids)>1:
            e=emb[ids]; sim=F.cosine_similarity(e.unsqueeze(1),e.unsqueeze(0),dim=2)
            m=torch.eye(len(ids))==0
            if m.any(): sc.append(sim[m].mean().item())
    return np.mean(sc) if sc else 0.0

# ── random baseline ──
with torch.no_grad():
    a0 = soft_assign(E, torch.randn(n_nodes,d), depth)
    l0 = a0.argmax(dim=1)
    g0 = leaf_groups(l0)
    t0 = [Counter(g).most_common(1)[0][0] if g else None for g in g0]

print(f"random: coh={coh(l0, E.detach()):.4f}  BLEU={sent_bleu(l0, t0):.2f}  "
      f"max_leaf={max(len(g) for g in g0)}  active={sum(1 for g in g0 if g)}")

# ── 训练 ──
opt = torch.optim.Adam([node_W, E], lr=0.03)
for step in range(500):
    opt.zero_grad()
    assign = soft_assign(E, node_W, depth)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)
    loss = F.mse_loss(E, lc[assign.argmax(dim=1)])
    ideal = V / n_leaves
    loss = loss + 3.0 * ((ls - ideal)**2).mean() / ideal  # 10x stronger balance
    loss.backward(); opt.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            a = soft_assign(E.detach(), node_W.detach(), depth)
            l = a.argmax(dim=1)
            g = leaf_groups(l)
            t = [Counter(g).most_common(1)[0][0] if g else None for g in g]
            c = coh(l, E.detach())
            b = sent_bleu(l, t)
            active = sum(1 for g in leaf_groups(l) if g)
        print(f"  step {step:3d}: loss={loss.item():.3f}  coh={c:.4f}  BLEU={b:.2f}  active={active}/{n_leaves}")

# ── 终验 ──
with torch.no_grad():
    Ef = E.detach(); nw = node_W.detach()
    a1 = soft_assign(Ef, nw, depth)
    l1 = a1.argmax(dim=1)
    g1 = leaf_groups(l1)
    t1 = [Counter(g).most_common(1)[0][0] if g else None for g in g1]

print(f"\ntrained: coh={coh(l1, Ef):.4f}  BLEU={sent_bleu(l1, t1):.2f}  "
      f"active={sum(1 for g in g1 if g)}")
for lid, words in sorted(enumerate(g1), key=lambda x: -len(x[1]))[:5]:
    ws = [id2word[w] for w in words[:5]]
    if len(words) > 0:
        print(f"  leaf {lid} ({len(words)}w): {ws}  top={id2word[Counter(words).most_common(1)[0][0]]}")
