"""
SPR Echo — Curriculum Learning: short→long sentences, shallow→deep tree
Compare: random shuffle vs progressive curriculum
"""
import torch, numpy as np, math, os, time
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
train_sents = load_sents(train_file, 30000)
val_sents = load_sents(val_file, 500)
print(f"train={len(train_sents)} val={len(val_sents)}")

# Vocab
word2id = {"<pad>": 0, "<unk>": 1}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v: k for k, v in word2id.items()}
full_depth = 14; n_leaves = 1 << full_depth
print(f"vocab={V} d={d} max_depth={full_depth} leaves={n_leaves}")

torch.manual_seed(42)
E = torch.randn(V, d).cuda()
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1)).cuda()[:d]

def spr_route(token_emb, depth):
    idx = 0; current = token_emb.clone()
    for dp in range(depth):
        current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK
        if torch.dot(token_emb, current) > 0:
            idx = 2 * idx + 2
        else:
            idx = 2 * idx + 1
    return idx - ((1<<depth) - 1)

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

def eval_bleu(leaf_for):
    lt = {}; lw = [[] for _ in range(n_leaves)]
    for wid in range(V): lw[leaf_for[wid].item()].append(wid)
    for lid in range(n_leaves):
        if lw[lid]: lt[lid] = lw[lid][0]
    refs, hyps = [], []
    for s in val_sents[:300]:
        ids = [word2id.get(w,1) for w in s]; 
        if len(ids)<4: continue
        refs.append(ids); hyps.append([lt.get(leaf_for[wid].item(),1) for wid in ids])
    return bleu(refs, hyps)

# ── Baseline: full depth, no curriculum ──
print("\n=== Baseline: full depth, no curriculum ===")
leaf_for = torch.zeros(V, dtype=torch.long)
for wid in range(V): leaf_for[wid] = spr_route(E[wid], full_depth)
solo = sum(1 for l in range(n_leaves) if (leaf_for==l).sum().item()==1)
b_base = eval_bleu(leaf_for)
print(f"  solo={solo}/{V} ({100*solo/V:.1f}%) BLEU={b_base:.2f}")

# ── Curriculum: progressive depth ──
print("\n=== Curriculum: progressive depth ===")
# Sort sentences by length
train_by_len = sorted(train_sents, key=len)

# Progressive depth: start at depth=4, gradually unlock
# Use a simple approach: for words appearing first in short sentences, 
# they get routed through the early depths first
# Simulate by: routing at different depths and merging results
depths = [(4, 0.0, 0.3), (8, 0.3, 0.6), (full_depth, 0.6, 1.0)]

leaf_curriculum = torch.zeros(V, dtype=torch.long)
# Count how many times each word appears in each length bucket
word_buckets = [Counter() for _ in range(3)]
for s in train_by_len:
    l = len(s)
    bucket = 0 if l <= 4 else (1 if l <= 8 else 2)
    for w in s:
        if w in word2id: word_buckets[bucket][word2id[w]] += 1

# For each word, determine which depth to use based on where it appears most
word_depth = torch.zeros(V, dtype=torch.long) + full_depth
for wid in range(V):
    max_bucket = max(range(3), key=lambda b: word_buckets[b].get(wid, 0))
    word_depth[wid] = depths[max_bucket][0]  # use the depth from that bucket

# Route words at their appropriate depth
for wid in range(V):
    leaf_curriculum[wid] = spr_route(E[wid], max(full_depth, int(word_depth[wid].item())))

# Actually the above assigns same depth to all. Let me use progressive routing:
# Words from short sentences get priority in shallow depths
# Then deep routing resolves collisions
# Simple implementation: route at full depth but weight by curriculum position

# Clearer approach: progressively grow the tree depth
# Stage 1 (0-30%): words mostly from short sentences
# Stage 2 (30-60%): words from medium sentences
# Stage 3 (60-100%): all words

# For SPR self-routing, the "curriculum" means: 
# Start with shallow depth, compute leaf assignments for short-sentence words
# Then deepen, recompute for remaining words
# This simulates what training would achieve

def curriculum_routing():
    """Progressive depth routing."""
    leaf = torch.zeros(V, dtype=torch.long)
    
    # Stage 1: depth=4, only route short-sentence words
    short_mask = torch.tensor([word_buckets[0].get(wid, 0) > word_buckets[1].get(wid, 0) + word_buckets[2].get(wid, 0) 
                               for wid in range(V)])
    for wid in range(V):
        if short_mask[wid]:
            leaf[wid] = spr_route(E[wid], 4)
    
    # Stage 2: depth=8, medium-sentence words
    med_mask = torch.tensor([word_buckets[1].get(wid, 0) >= word_buckets[0].get(wid, 0) and 
                             word_buckets[1].get(wid, 0) > word_buckets[2].get(wid, 0)
                             for wid in range(V)])
    for wid in range(V):
        if med_mask[wid] or leaf[wid] < 0:  # route medium + any overflow
            leaf[wid] = spr_route(E[wid], 8)
    
    # Stage 3: full depth for long-sentence words
    for wid in range(V):
        if not short_mask[wid] and not med_mask[wid]:
            leaf[wid] = spr_route(E[wid], full_depth)
        # Also reroute any that got bad assignments
        if leaf[wid] < 0:
            leaf[wid] = spr_route(E[wid], full_depth)
    
    return leaf

leaf_cur = curriculum_routing()
solo_cur = sum(1 for l in range(n_leaves) if (leaf_cur==l).sum().item()==1)
b_cur = eval_bleu(leaf_cur)
print(f"  solo={solo_cur}/{V} ({100*solo_cur/V:.1f}%) BLEU={b_cur:.2f}")
print(f"  delta: BLEU {b_cur - b_base:+.2f}")
