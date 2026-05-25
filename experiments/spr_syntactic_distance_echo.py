"""
SPR Syntactic Distance Tree — Pure geometry, zero parameters
1. Compute pairwise distance between adjacent token embeddings
2. Recursively split sentence at max distance → binary tree
3. Bottom-up merge via roll + SIGN_MASK → root hash
4. Root hash uniqueness test → can it serve as sentence signature?
"""
import torch, numpy as np, math
from collections import Counter, defaultdict

device = 'cuda'

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
print(f"train={len(train_sents)} val={len(val_sents)}")

word2id = {"<pad>": 0, "<unk>": 1}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
chunk_d = 16; chunk_depth = 7; chunk_leaves = 1 << chunk_depth
K = d // chunk_d  # K=4
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d} K={K} depth={chunk_depth} leaves={chunk_leaves} total={chunk_leaves**K:,}")

torch.manual_seed(42)
E = torch.randn(V, d, device=device)
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]
SIGN_MASK_CHUNK = torch.tensor([1., -1.] * (chunk_d//2 + 1), device=device)[:chunk_d]

# ──── Syntactic Distance ────
def syntactic_distances(embs, chunk_d, chunk_depth, sign_mask):
    """
    Compute d_i between each adjacent pair (i, i+1)
    Uses SPR chunk routing score magnitude as distance.
    Score = (chunk_emb * sign_alt(roll(chunk_emb, dp))).sum()
    Larger |score| = larger syntactic distance
    """
    T, d = embs.shape
    dists = torch.zeros(T-1, device=device)
    
    for k in range(d // chunk_d):
        chunk = embs[:, k*chunk_d:(k+1)*chunk_d]  # [T, chunk_d]
        current = chunk.clone()
        for dp in range(chunk_depth):
            current = torch.roll(current, shifts=dp+1, dims=-1) * sign_mask
            scores = (chunk * current).sum(dim=-1)  # [T]
            # Pairwise distance: |score[i] - score[i+1]| averaged across all chunks and depths
            dists += (scores[:-1] - scores[1:]).abs()
    
    return dists / (len(range(d // chunk_d)) * chunk_depth)  # normalize


# ──── Tree Building (recursive, bottom-up) ────
def build_tree(embs, dists):
    """
    Recursively split sentence at max distance.
    Returns: root_emb, tree_structure (pre-order splits)
    """
    T = len(embs)
    if T <= 1:
        return embs[0] if T == 1 else torch.zeros(d, device=device), []
    
    if len(dists) == 0:
        # Only 1 token, no distances
        return embs[0], []
    
    # Find split point: index of max distance
    split_pos = dists.argmax().item()
    
    # Build left and right subtrees
    left_embs = embs[:split_pos+1]
    left_dists = dists[:split_pos] if split_pos > 0 else dists[:0]
    right_embs = embs[split_pos+1:]
    right_dists = dists[split_pos+1:] if split_pos+1 < len(dists) else dists[:0]
    
    left_hash, left_struct = build_tree(left_embs, left_dists)
    right_hash, right_struct = build_tree(right_embs, right_dists)
    
    # Merge: SPR's roll + SIGN_MASK
    # Each level in the tree corresponds to a different shift
    depth_shift = 1  # simple for now
    merged = left_hash + SIGN_MASK * torch.roll(right_hash, shifts=depth_shift)
    merged = merged / (merged.norm() + 1e-8)
    
    # Tree structure: (split_pos, left_struct, right_struct)
    tree = [(split_pos, left_struct, right_struct)]
    
    return merged, tree

def compute_root_hash(ids):
    """Compute root hash for a sentence using syntactic distance tree"""
    if len(ids) < 2:
        return E[ids[0]] if len(ids) == 1 else torch.zeros(d, device=device)
    
    embs = E[torch.tensor(ids, device=device)]  # [T, d]
    dists = syntactic_distances(embs, chunk_d, chunk_depth, SIGN_MASK_CHUNK)
    root_hash, tree = build_tree(embs, dists)
    return root_hash


# ──── Root Hash Uniqueness Test ────
print(f"\n{'='*60}")
print("Building root hash table for training sentences...")
print(f"{'='*60}")

import time
t0 = time.time()

root_hash_table = {}  # root_hash_discrete → sentence tokens
collision_count = 0
total_sents = 0

for si, s in enumerate(train_sents[:10000]):
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    
    root_hash = compute_root_hash(ids)
    
    # Discretize root hash: which "leaf" would this vector land in?
    # Use same K-chunk routing as word-level SPR
    leaf_id = 0
    for k in range(K-1, -1, -1):
        chunk = root_hash[k*chunk_d:(k+1)*chunk_d]
        idx = 0
        current = chunk.clone()
        for dp in range(chunk_depth):
            current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK_CHUNK
            score = (chunk * current).sum()
            go_right = score > 0
            idx = idx * 2 + 1
            if go_right: idx += 1
        leaf_id = leaf_id * chunk_leaves + (idx - (chunk_leaves - 1))
    
    key = leaf_id
    
    if key in root_hash_table:
        # Collision! Another sentence has the same root hash
        prev_sent = root_hash_table[key]
        if prev_sent != tuple(ids):  # truly different sentence
            collision_count += 1
    else:
        root_hash_table[key] = tuple(ids)
    
    total_sents += 1
    
    if si % 2000 == 1999:
        elapsed = time.time() - t0
        print(f"  {si+1} sentences, {len(root_hash_table)} unique leaves, "
              f"{collision_count} collisions, {elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\nTotal: {total_sents} sentences")
print(f"Unique root hashes: {len(root_hash_table)}")
print(f"Collisions: {collision_count}/{total_sents} ({100*collision_count/total_sents:.2f}%)")
print(f"Time: {elapsed:.0f}s")

# ──── Echo BLEU Test ────
def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
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

print(f"\n{'='*60}")
print("Echo test: retrieve sentences by root hash")
print(f"{'='*60}")

refs, hyps = [], []
miss_count = 0

for s in val_sents[:300]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 2: continue
    
    root_hash = compute_root_hash(ids)
    
    leaf_id = 0
    for k in range(K-1, -1, -1):
        chunk = root_hash[k*chunk_d:(k+1)*chunk_d]
        idx = 0
        current = chunk.clone()
        for dp in range(chunk_depth):
            current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK_CHUNK
            score = (chunk * current).sum()
            go_right = score > 0
            idx = idx * 2 + 1
            if go_right: idx += 1
        leaf_id = leaf_id * chunk_leaves + (idx - (chunk_leaves - 1))
    
    key = leaf_id
    
    if key in root_hash_table:
        hyp = list(root_hash_table[key])
    else:
        hyp = [1] * len(ids)  # <unk>
        miss_count += 1
    
    refs.append(ids)
    hyps.append(hyp)

b = bleu(refs, hyps)
print(f"\nEcho BLEU-4 = {b:.2f}")
print(f"Val sentences: {len(refs)}")
print(f"Root hash not found (unseen): {miss_count}")

# ──── Samples ────
print(f"\n=== echo samples ===")
import random
random.seed(42)
samples = random.sample([s for s in train_sents[:10000] if len(s) >= 3], 5)
for s in samples:
    ids = [word2id.get(w, 1) for w in s]
    root_hash = compute_root_hash(ids)
    leaf_id = 0
    for k in range(K-1, -1, -1):
        chunk = root_hash[k*chunk_d:(k+1)*chunk_d]
        idx = 0
        current = chunk.clone()
        for dp in range(chunk_depth):
            current = torch.roll(current, shifts=dp+1, dims=-1) * SIGN_MASK_CHUNK
            score = (chunk * current).sum()
            go_right = score > 0
            idx = idx * 2 + 1
            if go_right: idx += 1
        leaf_id = leaf_id * chunk_leaves + (idx - (chunk_leaves - 1))
    key = leaf_id
    hyp = list(root_hash_table.get(key, [1]))
    src = ' '.join(id2word.get(w, '?') for w in ids[:8])
    hyp_str = ' '.join(id2word.get(w, '?') for w in hyp[:8])
    print(f"  src: {src}")
    print(f"  hyp: {hyp_str}")
    print()
