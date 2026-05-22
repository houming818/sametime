"""
SPR P8 — sentence-level shared tree + E^T decode
echo: encode → tree → H_leaf @ E^T → output token → BLEU
"""
import torch, numpy as np, math, os
from collections import Counter, defaultdict

tab = "/data/datasets/wmt14/wmt14.validation.de-en"
raw = []
with open(tab) as f:
    for l in f:
        if "\t" in l and len(raw) < 4:
            raw.append(l.split("\t", 1)[1].strip().lower().split())

# pad to 8
PAD = 8
word2id = {}
sent_tokens = []
for s in raw:
    ids = []
    for w in s[:PAD]:
        if w not in word2id: word2id[w] = len(word2id)
        ids.append(word2id[w])
    ids += [-1] * (PAD - len(ids))
    sent_tokens.append(ids)

V = len(word2id)
N_sents, N_tokens = len(sent_tokens), len(sent_tokens) * PAD
print(f"sents={N_sents} vocab={V} tokens_per_sent={PAD} total={N_tokens}")

torch.manual_seed(42)
E = torch.randn(V, 16) * 0.5  # shared embedding

def decode(leaf_vec):
    """leaf hash vector → nearest word embedding"""
    scores = leaf_vec @ E.T  # (V,)
    return scores.argmax().item()

def build_and_test(depth):
    n_leaves = 1 << depth
    n_nodes = n_leaves - 1
    node_W = torch.randn(n_nodes, 16)
    
    # build tree from ALL tokens across ALL sentences
    all_flat = []
    sent_map = []  # (sent_idx, pos_in_sent) for each token
    for si, s_ids in enumerate(sent_tokens):
        for pi, tid in enumerate(s_ids):
            all_flat.append((tid, si, pi))
    
    def recursively_split(items, node_idx, depth_remaining):
        if depth_remaining == 0 or len(items) <= 1:
            return {node_idx: items}
        real = [(t, s, p) for (t, s, p) in items if t >= 0]
        empt = [(t, s, p) for (t, s, p) in items if t < 0]
        
        if len(real) == 0:
            return {node_idx: items}
        
        emb_real = E[[t for (t,_,_) in real]]
        scores = emb_real @ node_W[node_idx]
        median = scores.median()
        
        left_r = [real[i] for i in range(len(real)) if scores[i] <= median]
        right_r = [real[i] for i in range(len(real)) if scores[i] > median]
        
        mid = len(empt)//2
        left = left_r + empt[:mid]
        right = right_r + empt[mid:]
        
        res = {}
        if left: res.update(recursively_split(left, 2*node_idx+1, depth_remaining-1))
        if right: res.update(recursively_split(right, 2*node_idx+2, depth_remaining-1))
        return res
    
    leaf_items = recursively_split(all_flat, 0, depth)
    
    # compute leaf hash vectors
    leaf_hash = {}
    for leaf_idx, items in leaf_items.items():
        real = [t for (t,_,_) in items if t >= 0]
        if real:
            leaf_hash[leaf_idx] = E[real].mean(dim=0)
        else:
            leaf_hash[leaf_idx] = torch.zeros(16)
    
    # token → leaf mapping
    token_to_leaf = {}
    for leaf_idx, items in leaf_items.items():
        for (tid, si, pi) in items:
            token_to_leaf[(si, pi)] = leaf_idx
    
    # decode: for each sentence position, get leaf hash → decode to word
    refs, hyps = [], []
    leaf_word_cache = {}  # cache decoded output per leaf
    
    for si, s_ids in enumerate(sent_tokens):
        ref = [t for t in s_ids if t >= 0]
        hyp = [-1] * PAD
        for pi, tid in enumerate(s_ids):
            if tid < 0: continue
            leaf_idx = token_to_leaf.get((si, pi))
            if leaf_idx is None: continue
            if leaf_idx not in leaf_word_cache:
                leaf_word_cache[leaf_idx] = decode(leaf_hash[leaf_idx])
            hyp[pi] = leaf_word_cache[leaf_idx]
        refs.append([r for r in ref if r >= 0])
        hyps.append([h for h in hyp if h >= 0])
    
    # BLEU
    def bleu(rf, hp):
        def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
        ps=[]
        for n in range(1,5):
            mch,ttl=0,0
            for r,h in zip(rf,hp):
                rc=Counter(ng(r,n)); hc=Counter(ng(h,n))
                ttl+=sum(hc.values()); mch+=sum(min(hc[k],rc.get(k,0)) for k in hc)
            ps.append(mch/max(ttl,1))
        bpv=[1-len(r)/max(len(h),1) for r,h in zip(rf,hp) if len(h)>0]
        bp=min(1.0,math.exp(max(bpv) if bpv else 0))
        return bp*math.exp(sum(math.log(max(p,1e-10)) for p in ps)/4)*100
    
    b = bleu(refs, hyps)
    
    # stats
    leaf_sizes = [len([t for (t,_,_) in items if t>=0]) for items in leaf_items.values()]
    solo = sum(1 for s in leaf_sizes if s==1)
    multi = sum(1 for s in leaf_sizes if s>1)
    
    return b, len(leaf_items), solo, multi, np.mean(leaf_sizes) if leaf_sizes else 0

print(f"\ndepth | leaves | avg tokens/leaf | solo | multi | BLEU-4")
print("-" * 60)
for depth in [3, 4, 5]:
    b, active, solo, multi, avg = build_and_test(depth)
    print(f"  {depth:2d}   | {active:5d} |     {avg:4.1f}         | {solo:4d} | {multi:4d} | {b:6.2f}")
