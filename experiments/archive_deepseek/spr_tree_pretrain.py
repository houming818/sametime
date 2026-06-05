"""
SPR Tree-Contrastive Pretraining — No external labels, no co-occurrence matrix
The tree structure ITSELF supervises E:
  - Words in same subtree → pull embeddings together
  - Words in different subtrees → push embeddings apart
After convergence, tree paths carry syntactic meaning.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")

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
train_sents = load_sents(train_file, 20000)
val_sents = load_sents(val_file, 500)

word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for s in train_sents:
    for w in s: freq[w] += 1
for w, c in freq.most_common():
    if c >= 2: word2id[w] = len(word2id)
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 128
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d} train={len(train_sents)} val={len(val_sents)}")

torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]

# ──── Syntactic Distance Tree ────
def syntactic_distances(embs):
    T = embs.shape[0]
    dists = torch.zeros(T-1, device=device)
    current = embs.clone()
    for dp in range(1, 4):
        rolled = torch.roll(current, shifts=dp, dims=-1) * SIGN_MASK
        scores = (embs * rolled).sum(dim=-1)
        dists += (scores[:-1] - scores[1:]).abs()
    return dists / 3.0

def build_tree_with_clusters(embs, dists):
    """
    Build tree. At each internal node, return the set of indices (word positions)
    that belong to this subtree. Used for contrastive sampling.
    Returns: list of (left_cluster, right_cluster) pairs at each split
    """
    T = len(embs)
    if T <= 1:
        return [list(range(len(embs)))]
    if len(dists) == 0:
        return [list(range(len(embs)))]
    
    split_pos = dists.argmax().item(); L = split_pos + 1
    
    # Recurse: get clusters from left and right subtrees
    left_clusters = build_tree_with_clusters(embs[:L], dists[:split_pos] if split_pos > 0 else dists[:0])
    right_clusters = build_tree_with_clusters(embs[L:], dists[L:] if L < len(dists) else dists[:0])
    
    # Adjust right cluster indices (shift by L)
    right_clusters = [[x+L for x in c] for c in right_clusters]
    
    # This node's cluster = union of all leaves
    all_leaves = sum(left_clusters, []) + sum(right_clusters, [])
    
    return [all_leaves] + left_clusters + right_clusters


# ──── Tree-Contrastive Pretraining ────
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
opt = torch.optim.Adam(E.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

print(f"\n{'='*60}")
print("Tree-Contrastive Pretraining of E")
print(f"  epochs=100 lr=0.01 margin=0.5")
t0 = time.time()

# Training subset
pretrain_sents = train_sents[:500]  # small subset for fast test

for epoch in range(30):
    E.train()
    random.shuffle(pretrain_sents)
    total_loss, total_n = 0, 0
    
    for bi in range(0, min(500, len(pretrain_sents)), 32):
        batch = pretrain_sents[bi:bi+32]
        if not batch: continue
        
        opt.zero_grad()
        loss_diff = torch.tensor(0.0, device=device)
        n = 0
        
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 4: continue
            
            ids_t = torch.tensor(ids, device=device)
            embs = E(ids_t)
            dists = syntactic_distances(embs)
            
            with torch.no_grad():
                clusters = build_tree_with_clusters(embs, dists)
            
            if not clusters: continue
            
            for c in clusters:
                if len(c) < 2: continue
                idx = random.sample(c, min(10, len(c)))
                for i in range(len(idx)):
                    for j in range(i+1, len(idx)):
                        cos_pos = F.cosine_similarity(
                            E(ids_t[idx[i]]).unsqueeze(0),
                            E(ids_t[idx[j]]).unsqueeze(0)
                        )
                        loss_diff += (1.0 - cos_pos).mean()
                        n += 1
            
            top_split = None
            for c in clusters:
                if len(c) >= 2:
                    if top_split is None: top_split = c
                    else:
                        s1 = random.sample(top_split, min(5, len(top_split)))
                        s2 = random.sample(c, min(5, len(c)))
                        for a in s1:
                            for b in s2:
                                cos_neg = F.cosine_similarity(
                                    E(ids_t[a]).unsqueeze(0),
                                    E(ids_t[b]).unsqueeze(0)
                                )
                                loss_diff += F.relu(cos_neg - 0.3).mean()
                                n += 1
                        break
        
        if n == 0: continue
        loss = loss_diff / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(E.parameters(), 2.0)
        opt.step()
        total_loss += loss.item()
        total_n += 1
    
    scheduler.step()
    
    if epoch % 5 == 0 or epoch == 29:
        elapsed = time.time() - t0
        
        # Check: for a few sample sentences, are tree paths meaningful?
        sample_sents = pretrain_sents[:5]
        print(f"  ep {epoch:3d} loss={total_loss/max(total_n,1):.4f} time={elapsed:.0f}s")
        
        for s in sample_sents[:2]:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 4: continue
            ids_t = torch.tensor(ids, device=device)
            with torch.no_grad():
                dists = syntactic_distances(E(ids_t))
                clusters = build_tree_with_clusters(E(ids_t), dists)
            # Show top-level split: which words go left, which go right
            if len(clusters) >= 2 and len(clusters[0]) >= 1:
                c0_words = [id2word.get(ids[i], '?') for i in clusters[0] if i < len(ids)]
                # Find the complement cluster (all indices not in clusters[0])
                all_used = set(c0_words)
                rest = [id2word.get(ids[i], '?') for i in range(len(ids)) if id2word.get(ids[i],'?') not in all_used]
                print(f"    {' '.join(s[:6])}:  LEFT=[{' '.join(c0_words[:4])}]  RIGHT=[{' '.join(rest[:4])}]")

# Save pretrained E
pretrained_E = E.weight.data.clone()
torch.save(pretrained_E.cpu(), '/tmp/spr_tree_pretrained_E.pt')
print(f"\nPretraining complete. E saved to /tmp/spr_tree_pretrained_E.pt")
print(f"Time={time.time()-t0:.0f}s")

# ──── Visualize: tree path patterns ────
print(f"\n{'='*60}")
print(f"Tree path pattern analysis (post-pretraining)")
print(f"{'='*60}")

path_counter = Counter()
for s in pretrain_sents[:200]:
    ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 4: continue
    ids_t = torch.tensor(ids, device=device)
    with torch.no_grad():
        dists = syntactic_distances(E(ids_t))
        # Build full paths
        def get_paths(embs, d, prefix=''):
            T = len(embs)
            if T <= 1: return [prefix]
            if len(d) == 0: return [prefix]
            split = d.argmax().item(); L = split + 1
            return get_paths(embs[:L], d[:split] if split>0 else d[:0], prefix+'L') + \
                   get_paths(embs[L:], d[L:] if L<len(d) else d[:0], prefix+'R')
        paths = get_paths(E(ids_t), dists)
        for pth, tok_id in zip(paths, ids):
            word = id2word.get(tok_id, '?')
            path_counter[(pth, word)] += 1

# Show: for each path prefix, what words appear there?
path_words = defaultdict(Counter)
for (pth, word), cnt in path_counter.items():
    path_words[pth[:3]][word] += cnt  # group by first 3 levels of path

print("Top 5 words per path prefix (first 3 levels):")
for pth, wc in sorted(path_words.items(), key=lambda x: -sum(x[1].values()))[:10]:
    top_words = wc.most_common(5)
    print(f"  {pth}: {', '.join(f'{w}({c})' for w,c in top_words)}")
