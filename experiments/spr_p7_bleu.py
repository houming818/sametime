"""
P7 BLEU v6 — per-node hard tree (2^depth-1 nodes, each with own weight)
hard routing, no training needed — tree built once top-down
"""
import torch
import numpy as np, math, os
from collections import Counter

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
word2id, id2word, sents = {}, {}, []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(sents) < 500:
            en = l.split("\t", 1)[1].strip().lower()
            words = en.split(); sents.append(words)
            for w in words:
                if w not in word2id: word2id[w] = len(word2id); id2word[len(id2word)] = w

V, d = len(word2id), 16
depth = 1
while 2**depth < V: depth += 1
n_nodes = (1 << depth) - 1
n_leaves = 1 << depth

print(f"vocab={V} depth={depth} nodes={n_nodes} leaves={n_leaves} params={n_nodes}×{d}={n_nodes*d}")

torch.manual_seed(42)
emb = torch.randn(V, d) * 0.5

# per-node random projection
node_W = torch.randn(n_nodes, d)
leaf_words = [[] for _ in range(n_leaves)]

def build_tree(word_ids, node_idx, depth):
    """Hard split by sign(token @ node_W[node_idx])"""
    if depth == 0 or len(word_ids) <= 1:
        leaf_idx = node_idx - (2**(depth_orig - depth) - 1)
        # FIX: need depth tracking
        leaf_words[leaf_idx].extend(word_ids)
        return
    tokens = emb[word_ids]
    scores = tokens @ node_W[node_idx]
    left = word_ids[scores <= 0]
    right = word_ids[scores > 0]
    if len(left) > 0:
        build_tree(left, 2*node_idx+1, depth-1)
    if len(right) > 0:
        build_tree(right, 2*node_idx+2, depth-1)

# Track leaf mapping
leaf_for_word = torch.zeros(V, dtype=torch.long)
leaf_assigned = [[] for _ in range(n_leaves)]

def build_tree_v2(word_ids, node_idx, depth):
    """Build tree, track leaf for each word, assign leaf slots"""
    if len(word_ids) == 0:
        return
    scores = emb[word_ids] @ node_W[node_idx]
    median = scores.median()
    left = word_ids[scores <= median]
    right = word_ids[scores > median]
    
    if depth == 1:
        # Leaf level — assign left and right to specific leaves
        for wid in left: leaf_for_word[wid] = 2*node_idx + 1 - n_nodes
        for wid in right: leaf_for_word[wid] = 2*node_idx + 2 - n_nodes
        leaf_assigned[2*node_idx+1-n_nodes].extend(left.tolist())
        leaf_assigned[2*node_idx+2-n_nodes].extend(right.tolist())
        return
    
    if len(left) > 0: build_tree_v2(left, 2*node_idx+1, depth-1)
    if len(right) > 0: build_tree_v2(right, 2*node_idx+2, depth-1)

full_depth = depth
build_tree_v2(torch.arange(V), 0, full_depth)

# each leaf's top word
leaf_top = {}
for lid in range(n_leaves):
    if leaf_assigned[lid]:
        leaf_top[lid] = Counter(leaf_assigned[lid]).most_common(1)[0][0]

active = sum(1 for lw in leaf_assigned if lw)
avg = np.mean([len(lw) for lw in leaf_assigned if lw])
solo = sum(1 for lw in leaf_assigned if len(lw) == 1)
dead = sum(1 for lw in leaf_assigned if len(lw) == 0)

print(f"active={active}/{n_leaves}  avg words/leaf={avg:.1f}  solo={solo}  dead={dead}")

# BLEU
def bleu(refs, hyps):
    def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    ps = []
    for n in range(1,5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = Counter(ng(r, n)); hc = Counter(ng(h, n))
            ttl += sum(hc.values())
            mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1))
    bpv = [1 - len(r)/max(len(h),1) for r,h in zip(refs, hyps) if len(h)>0]
    bp = min(1.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100

refs, hyps = [], []
for s in sents:
    rids = [word2id[w] for w in s]
    pids = [leaf_top[leaf_for_word[wid].item()] for wid in rids]
    refs.append(rids); hyps.append(pids)

b = bleu(refs, hyps)
print(f"BLEU-4 = {b:.2f}")
