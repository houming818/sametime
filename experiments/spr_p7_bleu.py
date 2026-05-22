"""
P7 BLEU — WMT14 真实词表
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, os
from collections import Counter

# ── 读 WMT14 词表 ──
tab = "/data/datasets/wmt14/wmt14.validation.de-en"

word2id, id2word, sents = {}, {}, []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(sents) < 500:
            en = l.split("\t", 1)[1].strip().lower()
            words = en.split()
            sents.append(words)
            for w in words:
                if w not in word2id:
                    idx = len(word2id); word2id[w] = idx; id2word[idx] = w

V, d = len(word2id), 32

# 树深度：确保每词独叶
depth = 1
while 2**depth < V:
    depth += 1
n_leaves = 2**depth

print(f"WMT14 en vocab: {V} words, depth={depth}, leaves={n_leaves}")

# ── embedding + 树 ──
torch.manual_seed(42)
embed = torch.randn(V, d) * 0.5
weights = nn.ParameterList([nn.Parameter(torch.randn(d)*0.05) for _ in range(depth)])
opt = torch.optim.Adam(weights, lr=0.03)

def route(tokens, ws, dpt):
    if dpt == 1:
        pr = torch.sigmoid(tokens @ ws[0])
        return torch.cat([(1-pr).unsqueeze(1), pr.unsqueeze(1)], dim=1)
    pr = torch.sigmoid(tokens @ ws[0])
    child = route(tokens, ws[1:], dpt-1)
    L = child.shape[1]
    a = torch.zeros(len(tokens), L*2)
    a[:,:L] = (1-pr).unsqueeze(1) * child
    a[:,L:] = pr.unsqueeze(1) * child
    return a

all_ids = list(range(V))
all_emb = embed[all_ids]
for step in range(200):
    opt.zero_grad(); assign = route(all_emb, weights, depth)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ all_emb / ls.unsqueeze(1)
    # smaller vocab: full distance matrix is fine
    dist = torch.cdist(all_emb, lc)
    with torch.no_grad():
        best = dist.argmin(dim=1)
        target = torch.zeros_like(assign)
        target[range(V), best] = 1.0
    loss = F.mse_loss(assign, target)
    loss.backward(); opt.step()

with torch.no_grad():
    assign_f = route(all_emb, [w.detach() for w in weights], depth)
    leaf_for = assign_f.argmax(dim=1)
    leaf_words = [[] for _ in range(n_leaves)]
    for wid in list(range(V)):
        leaf_words[leaf_for[wid].item()].append(wid)
    leaf_top = {}
    for lid in list(range(n_leaves)):
        if leaf_words[lid]:
            leaf_top[lid] = Counter(leaf_words[lid]).most_common(1)[0][0]
    active = sum(1 for lw in leaf_words if lw)
    avg = np.mean([len(lw) for lw in leaf_words if lw])
    solo = sum(1 for lw in leaf_words if len(lw) == 1)
    print(f"trained: active={active}/{n_leaves}  avg words/leaf={avg:.1f}  solo leaves={solo}")
    print(f"training loss: {loss.item():.4f}")

# ── BLEU ──
def bleu(refs, hyps):
    def ng(tokens, n): return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = Counter(ng(r, n)); hc = Counter(ng(h, n))
            ttl += sum(hc.values())
            mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1))
    bpv = [1 - len(r)/max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    bp = min(1.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

refs, hyps = [], []
for s in sents:
    rids = [word2id[w] for w in s]
    pids = [leaf_top[leaf_for[wid].item()] for wid in rids]
    refs.append(rids); hyps.append(pids)

b = bleu(refs, hyps)
print(f"BLEU-4 = {b:.2f}")
