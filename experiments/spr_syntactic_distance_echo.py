"""
SPR Syntactic Distance Tree — Flat Path-Conditioned Decoder (v5)
Fixes:
 1. zero_grad/step moved OUTSIDE batch loop → true batch accumulation
 2. Separate L/R path embeddings → no sign-cancellation collapse
 3. E included in Adam → word embeddings co-evolve with decoder
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
    left_root, left_leaves, left_paths = build_tree_with_paths(
        embs[:L], dists[:split_pos] if split_pos > 0 else dists[:0], tree_depth + 1)
    right_root, right_leaves, right_paths = build_tree_with_paths(
        embs[L:], dists[L:] if L < len(dists) else dists[:0], tree_depth + 1)
    merged = left_root + SIGN_MASK * torch.roll(right_root, shifts=tree_depth)
    merged = merged / (merged.norm() + 1e-8)
    return merged, left_leaves + right_leaves, ['L' + p for p in left_paths] + ['R' + p for p in right_paths]


class FlatPathDecoder(nn.Module):
    """Fix #2: separate L/R embeddings — no sign-flip cancellation"""
    def __init__(self, d, max_path_len=32):
        super().__init__()
        self.d = d
        self.path_enc_L = nn.Embedding(max_path_len, d)
        self.path_enc_R = nn.Embedding(max_path_len, d)
        self.decoder = nn.Sequential(
            nn.Linear(d * 2, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d)
        )
    
    def forward_sentence(self, root_hash, leaf_paths):
        T = len(leaf_paths)
        if T == 0:
            return torch.zeros(0, self.d, device=root_hash.device)
        
        path_vecs = []
        for path in leaf_paths:
            v = torch.zeros(self.d, device=root_hash.device)
            for depth, c in enumerate(path[:self.path_enc_L.num_embeddings]):
                idx = torch.tensor(depth, device=root_hash.device)
                pe = self.path_enc_L(idx) if c == 'L' else self.path_enc_R(idx)
                v = v + pe
            path_vecs.append(v / max(len(path), 1))
        
        path_tensor = torch.stack(path_vecs)  # [T, d]
        root_tensor = root_hash.unsqueeze(0).expand(T, -1)  # [T, d]
        inp = torch.cat([root_tensor, path_tensor], dim=-1)  # [T, 2*d]
        return self.decoder(inp)  # [T, d]


decoder = FlatPathDecoder(d).to(device)
# Fix #3: E.parameters() in optimizer
opt = torch.optim.Adam(list(decoder.parameters()) + list(E.parameters()), lr=0.002)
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
print(f"Training Flat-Path Decoder v5 (E in opt, batch accum, L/R path enc)")
print(f"  epochs=100 batch=16 lr=0.002")
t0 = time.time()

for epoch in range(100):
    decoder.train(); E.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    for bi in range(0, 2000, 16):
        batch_sents = train_sents[bi:bi+16]
        if not batch_sents: continue
        
        # Fix #1: accumulate gradients across batch
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        n_tokens = 0
        
        for s in batch_sents:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 3: continue
            ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
            
            with torch.no_grad():
                dists = syntactic_distances(embs)
                root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
            
            leaf_hashes = decoder.forward_sentence(root_hash, leaf_paths)
            
            logits = leaf_hashes @ E.weight.T  # [T, V]
            loss = F.cross_entropy(logits, ids_t)
            batch_loss = batch_loss + loss
            n_tokens += len(ids)
        
        if n_tokens == 0: continue
        batch_loss = batch_loss / n_tokens
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(E.parameters()), 2.0)
        opt.step()
        
        ti += 1; tl += batch_loss.item(); tt += 1
    
    if ti == 0: continue
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 99:
        decoder.eval(); E.eval()
        refs, hyps = [], []
        with torch.no_grad():
            for s, ids in val_data[:50]:
                ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
                dists = syntactic_distances(embs)
                root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
                leaf_hashes = decoder.forward_sentence(root_hash, leaf_paths)
                logits = leaf_hashes @ E.weight.T
                pred = logits.argmax(dim=-1).cpu().tolist()
                refs.append(ids); hyps.append(pred)
        bleu = compute_bleu(refs, hyps)
        tok_acc = sum(1 for r,h in zip(refs,hyps) for ri,hi in zip(r,h) if ri==hi)
        tok_tot = sum(len(r) for r in refs)
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} "
              f"tok_acc={tok_acc}/{tok_tot}={100*tok_acc/tok_tot:.1f}% "
              f"time={time.time()-t0:.0f}s")
        decoder.train(); E.train()

# Final
decoder.eval(); E.eval(); refs, hyps = [], []
with torch.no_grad():
    for s, ids in val_data:
        ids_t = torch.tensor(ids, device=device); embs = E(ids_t)
        dists = syntactic_distances(embs)
        root_hash, _, leaf_paths = build_tree_with_paths(embs, dists)
        leaf_hashes = decoder.forward_sentence(root_hash, leaf_paths)
        logits = leaf_hashes @ E.weight.T
        pred = logits.argmax(dim=-1).cpu().tolist()
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
    leaf_hashes = decoder.forward_sentence(root_hash, leaf_paths)
    logits = leaf_hashes @ E.weight.T
    pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:8])}")
    print(f"  hyp: {' '.join(pred[:8])}")
    print()
