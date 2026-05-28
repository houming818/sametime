"""
SPR Heap + Path Encoding Echo — Single complete binary tree
Pad to power-of-2 → each node has unique heap index → path_emb(node_idx)
Flat decoder: (root_hash, path_emb) → MLP → leaf → dot E → token
No templates, no GRU, no W_split recursion
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR HEAP+PATH ECHO")
print("=" * 60)

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
train_sents = load_sents(train_file, 100000)
val_sents = load_sents(val_file, 500)

word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for s in train_sents:
    for w in s: freq[w] += 1
for w, c in freq.most_common():
    if c >= 5: word2id[w] = len(word2id)
for s in val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 128
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d}")

# ──── Heap tree ────
def heap_size(T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    leaves = 1 << (k - 1)
    total = (1 << k) - 1
    return total, leaves, k

torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d // 2 + 1), device=device)[:d]

def heap_encode(E, ids):
    T = len(ids)
    P, n_leaves, depth = heap_size(T)
    ids_pad = ids + [0] * (P - T)
    ids_t = torch.tensor(ids_pad, device=device)
    embs = E(ids_t)  # [P, d]
    # Bottom-up merge (safe for-loop, O(logT) — negligible cost)
    for i in range(P - 1, 0, -2):
        parent = (i - 1) // 2
        if i < P:
            merged = embs[i - 1] + SIGN_MASK * torch.roll(embs[i], shifts=1)
            embs[parent] = merged / (merged.norm() + 1e-8)
    root = embs[0]
    leaf_start = (1 << (depth - 1)) - 1 if depth > 0 else 0
    mask = torch.tensor([1.0 if t < T else 0.0 for t in range(n_leaves)], device=device)
    return root, embs[leaf_start:leaf_start + n_leaves], mask, P, n_leaves, depth


# ──── Path Encoding ────
class PathEncoder(nn.Module):
    def __init__(self, d, max_heap=256):
        super().__init__()
        self.depth_enc = nn.Embedding(16, d)
        self.index_enc = nn.Embedding(max_heap, d)
    def forward(self, leaf_indices, P, tree_depth):
        """leaf_indices: [N] — batch of leaf heap indices. Returns [N, d]"""
        N = len(leaf_indices)
        leaf_level = tree_depth - 1
        level_start = (1 << leaf_level) - 1 if leaf_level >= 0 else 0
        rel_pos = leaf_indices - level_start
        max_nodes = max(1, 1 << leaf_level)
        d_idx = torch.clamp(torch.full((N,), leaf_level, device=device, dtype=torch.long), 0, 15)
        i_idx = torch.clamp(rel_pos % max_nodes, 0, self.index_enc.weight.shape[0] - 1)
        return self.depth_enc(d_idx) + self.index_enc(i_idx)


# ──── Flat Decoder (vectorized) ────
class FlatPathDecoder(nn.Module):
    def __init__(self, d, V, max_heap=256):
        super().__init__()
        self.d = d
        self.path_enc = PathEncoder(d, max_heap)
        self.mlp = nn.Sequential(nn.Linear(d * 2, d * 4), nn.ReLU(), nn.Linear(d * 4, d))
    def forward(self, root, E_weight, P, n_leaves, depth, T):
        # All leaves in one matrix operation
        leaf_indices = torch.arange(
            (1 << (depth - 1)) - 1 if depth > 0 else 0,
            (1 << (depth - 1)) - 1 + n_leaves if depth > 0 else n_leaves,
            device=device
        )  # [n_leaves]
        path_vecs = self.path_enc(leaf_indices, P, depth)  # [n_leaves, d]
        root_expand = root.unsqueeze(0).expand(n_leaves, -1)  # [n_leaves, d]
        inp = torch.cat([root_expand, path_vecs], dim=-1)  # [n_leaves, 2d]
        leaf_tensor = self.mlp(inp)  # [n_leaves, d]
        logits = leaf_tensor[:T] @ E_weight.T
        return logits, leaf_tensor


# ──── Init ────
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
decoder = FlatPathDecoder(d, V).to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
print(f"params={sum(p.numel() for m in [E,decoder] for p in m.parameters())/1e6:.1f}M")

def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def compute_bleu(refs, hyps):
    C = Counter; ps = []
    for n in range(1, 5):
        mch, ttl = 0, 0
        for r, h in zip(refs, hyps):
            rc = C(ng(r, n)); hc = C(ng(h, n))
            ttl += sum(hc.values()); mch += sum(min(hc[k], rc.get(k, 0)) for k in hc)
        ps.append(mch / max(ttl, 1) if ttl > 0 else 1.0)
    bpv = [1 - len(r) / max(len(h), 1) for r, h in zip(refs, hyps) if len(h) > 0]
    bp = min(2.0, math.exp(max(bpv) if bpv else 0))
    return bp * math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / 4) * 100

val_data = [(s, [word2id.get(w, 1) for w in s]) for s in val_sents[:100] if len(s) >= 2]

# ──── Train ────
print(f"\n{'='*60}")
print(f"Heap+Path Echo: pad→complete binary, path_emb(heap_index), flat MLP decoder")
print(f"  epochs=30 batch=16 lr=0.003")
t0 = time.time()

for epoch in range(30):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0

    for bi in range(0, 10000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            T = len(ids)
            with torch.no_grad():
                root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
            ids_t = torch.tensor(ids, device=device)
            logits, pred_leaves = decoder(root, E.weight, P, n_leaves, depth, T)
            loss = F.cross_entropy(logits[:T], ids_t) + F.mse_loss(pred_leaves[:T], gold_leaves[:T])
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0); opt.step()
        ti += 1; tl += (bl / n).item(); tt += 1

    if ti == 0: continue
    avg_loss = tl / ti

    if epoch % 5 == 0 or epoch == 29:
        E.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = len(ids)
                root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
                logits, _ = decoder(root, E.weight, P, n_leaves, depth, T)
                pred = logits[:T].argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T]); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        E_mu = E.weight.mean().item(); E_sigma = E.weight.std().item()
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={avg_loss:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% E μ={E_mu:.3f} σ={E_sigma:.3f} {elapsed:.0f}s")
        E.train(); decoder.train()

# Final
E.eval(); decoder.eval(); rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        T = len(ids)
        root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
        logits, _ = decoder(root, E.weight, P, n_leaves, depth, T)
        pred = logits[:T].argmax(dim=-1).cpu().tolist()
        rf.append(ids[:T]); hp.append(pred)
bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    T = len(ids)
    with torch.no_grad():
        root, _, _, P, n_leaves, depth = heap_encode(E, ids)
        logits, _ = decoder(root, E.weight, P, n_leaves, depth, T)
        pred = [id2word.get(p, '?') for p in logits[:T].argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}  [P={P}]")
    print()
