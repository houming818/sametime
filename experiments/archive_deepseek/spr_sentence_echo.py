"""
SPR Sentence-Level Echo S1 — Pure geometry, zero training
Positional sin/cos + causal momentum → each (word, position) gets unique leaf
"""
import torch, numpy as np, math, os
from collections import Counter, defaultdict

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

K = 4; chunk_d = d // K; chunk_depth = 7; chunk_leaves = 1 << chunk_depth
total_leaves = chunk_leaves ** K
SIGN_MASK = torch.tensor([1., -1.] * (chunk_d//2 + 1)).cuda()[:chunk_d]
SIGN_MASK_FULL = torch.tensor([1., -1.] * (d//2 + 1)).cuda()[:d]
print(f"K={K} chunk_leaves={chunk_leaves} total_leaves={total_leaves:,}")

# ── Sentence-level routing: pos_emb + causal momentum ──
def route_sentence(context_embs):
    """route_sentence_s1: positional + temporal context injection"""
    T = len(context_embs)
    
    # 1. Sin/cos positional encoding (same as Transformer, zero params)
    pos = torch.arange(T, device='cuda').unsqueeze(1).float()
    phase = pos / (10000 ** (torch.arange(0, d, 2, device='cuda').float() / d))
    pos_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    
    ctx = context_embs + 0.5 * pos_emb  # balance: word dominates routing, position separates
    ctx = ctx / (ctx.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 3. Route each position through K chunks
    leaf_chunks = torch.zeros(T, K, dtype=torch.long, device='cuda')
    for k in range(K):
        chunk = ctx[:, k*chunk_d:(k+1)*chunk_d]  # (T, 16)
        idx = torch.zeros(T, dtype=torch.long, device='cuda')
        current = chunk.clone()
        
        for dp in range(chunk_depth):
            current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK
            if T > 1:
                current = current + 0.1 * torch.roll(current, shifts=1, dims=0)  # neighbor mix
            scores = (chunk * current).sum(dim=-1)
            go_right = scores > 0
            idx = idx * 2 + 1
            idx[go_right] += 1
        
        leaf_chunks[:, k] = idx - (chunk_leaves - 1)
    
    # 4. Combine into 268M-bin space
    leaf_combined = torch.zeros(T, dtype=torch.long, device='cuda')
    for k in range(K-1, -1, -1):
        leaf_combined = leaf_combined * chunk_leaves + leaf_chunks[:, k]
    return leaf_combined

# ── Register all sentence tokens from training set ──
print("registering sentence tokens...")
leaf_top = {}
total_tokens = 0
solo_count = 0
collision_count = 0

for si, s in enumerate(train_sents[:10000]):
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    T = len(ids)
    
    embs = E[ids]  # (T, d)
    leaves = route_sentence(embs)  # (T,) — unique per (word, position)
    
    for t in range(T):
        lid = leaves[t].item()
        key = lid  # use leaf as key
        if key not in leaf_top:
            leaf_top[key] = ids[t]  # first word to land here
            solo_count += 1
        else:
            collision_count += 1
    total_tokens += T
    
    if si % 1000 == 999: print(f"  {si+1} sentences, {total_tokens} tokens")

print(f"registered: {total_tokens} tokens, {len(leaf_top)} unique leaves")
print(f"collisions: {collision_count}/{total_tokens} ({100*collision_count/total_tokens:.2f}%)")

# ── BLEU on validation ──
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
val_tokens = 0
val_recovered = 0

for s in val_sents[:200]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    T = len(ids)
    embs = E[ids]
    leaves = route_sentence(embs)
    
    hyp = [leaf_top.get(leaves[t].item(), 1) for t in range(T)]
    refs.append(ids)
    hyps.append(hyp)
    
    for t in range(T):
        if hyp[t] == ids[t]:
            val_recovered += 1
    val_tokens += T

b = bleu(refs, hyps)
print(f"\n=== Sentence-Level Echo BLEU-4 = {b:.2f} ===")
print(f"token recovery: {val_recovered}/{val_tokens} ({100*val_recovered/val_tokens:.2f}%)")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    embs = E[ids]
    leaves = route_sentence(embs)
    hyp = [id2word.get(leaf_top.get(leaves[t].item(), 1), '?') for t in range(len(ids))]
    src = ' '.join([id2word.get(w, '?') for w in ids[:6]])
    print(f"  src: {src}")
    print(f"  hyp: {' '.join(hyp[:6])}")
    print()
