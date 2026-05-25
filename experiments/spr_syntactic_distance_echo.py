"""
SPR Syntactic Distance Tree — Flat Path-Conditioned Decoder
Encode: syntactic distances → recursive tree → root_hash + leaf_paths
Decode: each leaf = MLP(root_hash + path_embedding), no recursion, no gradient chain
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

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
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
SIGN_MASK = torch.tensor([1., -1.] * (d//2 + 1), device=device)[:d]

def syntactic_distances(embs):
    T = embs.shape[0]
    dists = torch.zeros(T-1, device=device)
    current = embs.clone()
    for dp in range(1, 4):
        rolled = torch.roll(current, shifts=dp, dims=-1) * SIGN_MASK
        scores = (embs * rolled).sum(dim=-1)
        dists += (scores[:-1] - scores[1:]).abs()
    return dists / 3.0

def build_tree_with_paths(embs, dists, tree_depth=1):
    T = len(embs)
    if T <= 1:
        leaf = embs[0] if T >= 1 else torch.zeros(d, device=device)
        return leaf, [leaf], ['']
    if len(dists) == 0:
        return embs[0], [embs[0]], ['']
    split_pos = dists.argmax().item(); L = split_pos + 1
    left_root, left_leaves, left_paths = build_tree_with_paths(embs[:L], dists[:split_pos] if split_pos > 0 else dists[:0], tree_depth + 1)
    right_root, right_leaves, right_paths = build_tree_with_paths(embs[L:], dists[L:] if L < len(dists) else dists[:0], tree_depth + 1)
    merged = left_root + SIGN_MASK * torch.roll(right_root, shifts=tree_depth)
    merged = merged / (merged.norm() + 1e-8)
    return merged, left_leaves + right_leaves, ['L' + p for p in left_paths] + ['R' + p for p in right_paths]


class FlatPathDecoder(nn.Module):
    """Direct mapping: root_hash + path_info → leaf_hash (no recursion)"""
    def __init__(self, d, max_path_len=20):
        super().__init__()
        self.d = d
        # Path embedding: 'L'→[0], 'R'→[1], positional encoding for depth
        self.path_enc = nn.Embedding(max_path_len, d)  # position → embedding
        
        # MLP: [root_hash(d) + path_summary(d)] → leaf_hash(d)
        self.decoder = nn.Sequential(
            nn.Linear(d * 2, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d)
        )
    
    def forward(self, root_hash, leaf_paths):
        T = len(leaf_paths)
        leaf_hashes = torch.zeros(T, self.d, device=root_hash.device)
        
        for t, path in enumerate(leaf_paths):
            # Encode path: sum of position embeddings weighted by direction
            path_vec = torch.zeros(self.d, device=root_hash.device)
            for depth, c in enumerate(path):
                if depth < self.path_enc.num_embeddings:
                    pe = self.path_enc(torch.tensor(depth, device=root_hash.device))
                    if c == 'R': pe = -pe  # sign flip for right
                    path_vec = path_vec + pe
            path_vec = path_vec / max(len(path), 1)  # normalize
            
            # Combine root + path → leaf
            inp = torch.cat([root_hash, path_vec])
            leaf_hashes[t] = self.decoder(inp)
        
        return leaf_hashes


decoder = FlatPathDecoder(d).to(device)
opt = torch.optim.Adam(decoder.parameters(), lr=0.003)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values())
            mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bp = min(2.0, math.exp(max(0, max(1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0))))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents[:200] if len(s) >= 2]

print(f"\n{'='*60}")
print(f"Training Flat-Path Decoder...")
print(f"  epochs=100 batch=32 lr=0.003")
t0 = time.time()

for epoch in range(100):
    decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    for bi in range(0, 2000, 32):
        batch_sents = train_sents[bi:bi+32]
        if not batch_sents: continue
        for s in batch_sents:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
            with torch.no_grad():
                dists = syntactic_distances(embs)
                root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
            leaf_hashes = decoder(root_hash, leaf_paths)
            loss = torch.tensor(0.0, device=device)
            for t in range(len(ids)):
                logits = leaf_hashes[t] @ E.weight.T
                loss += F.cross_entropy(logits.unsqueeze(0), ids_t[t].unsqueeze(0))
            loss = loss / len(ids)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0); opt.step()
            ti += 1; tl += loss.item(); tt += len(ids)
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 99:
        decoder.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
                dists = syntactic_distances(embs)
                root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
                leaf_hashes = decoder(root_hash, leaf_paths)
                pred = [int((leaf_hashes[t] @ E.weight.T).argmax().item()) for t in range(len(ids))]
                refs.append(ids); hyps.append(pred)
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc}/{sum(len(r) for r in refs)}={100*tok_acc/sum(len(r) for r in refs):.1f}% time={time.time()-t0:.0f}s")
        decoder.train()

# Final
decoder.eval(); refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
        dists = syntactic_distances(embs)
        root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
        leaf_hashes = decoder(root_hash, leaf_paths)
        pred = [int((leaf_hashes[t] @ E.weight.T).argmax().item()) for t in range(len(ids))]
        refs.append(ids); hyps.append(pred)
print(f"\nFinal BLEU-4 = {compute_bleu(refs, hyps):.1f}")
print(f"Token accuracy = {sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)}/{sum(len(r) for r in refs)}")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
    dists = syntactic_distances(embs)
    root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
    leaf_hashes = decoder(root_hash, leaf_paths)
    pred = [id2word.get((leaf_hashes[t] @ E.weight.T).argmax().item(), '?') for t in range(len(ids))]
    print(f"  src: {' '.join(s[:8])}")
    print(f"  hyp: {' '.join(pred[:8])}")
    print()
