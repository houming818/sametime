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
train_sents = load_sents(train_file, 50000)
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
    ids = ids[:60]; T = len(ids)
    P, n_leaves, depth = heap_size(T)
    ids_pad = ids + [0] * (n_leaves - T)
    ids_t = torch.tensor(ids_pad, device=device)
    leaf_matrix = E(ids_t)  # [n_leaves, d] — zero compression
    mask = torch.tensor([1.0 if t < T else 0.0 for t in range(n_leaves)], device=device)
    return leaf_matrix, mask, n_leaves, depth


# ──── Path Encoding (RNN-style L/R accumulation) ────
class PathEncoder(nn.Module):
    def __init__(self, d, max_depth=12):
        super().__init__()
        self.depth_enc = nn.Embedding(max_depth, d)
        self.dir_L = nn.Embedding(max_depth, d)
        self.dir_R = nn.Embedding(max_depth, d)
    def forward(self, leaf_indices, tree_depth):
        """
        leaf_indices: [N] heap indices. tree_depth: total tree depth.
        Accumulate L/R decisions along binary path from root to leaf.
        Returns [N, d]
        """
        N = len(leaf_indices)
        emb = torch.zeros(N, self.depth_enc.weight.shape[1], device=device)
        # Convert heap index to binary path. For heap, root=1, left=2i, right=2i+1
        # For index 0-rooted: parent(i) = (i-1)//2, direction = i%2 (0=left, 1=right)
        for lvl in range(tree_depth):
            idx = torch.tensor(lvl, device=device)
            d_emb = self.depth_enc(idx).unsqueeze(0)  # [1,d]
            emb = emb + d_emb
            shift = tree_depth - 1 - lvl
            bits = (leaf_indices >> shift) & 1  # [N] — 0 or 1 at this depth
            for i, bit in enumerate(bits.cpu().tolist()):
                if bit == 0:
                    emb[i] = emb[i] + self.dir_L(idx)
                else:
                    emb[i] = emb[i] + self.dir_R(idx)
        return emb


# ──── Flat Decoder (vectorized) ────
class FlatPathDecoder(nn.Module):
    def __init__(self, d, n_splits=3, max_depth=12, max_leaves=64):
        super().__init__()
        self.d = d
        self.path_enc = PathEncoder(d, max_depth)
        # Selection MLP: (leaf_matrix[i] + path_emb[i]) → refine → leaf
        self.refine = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.max_leaves = max_leaves

    def forward(self, root, E_weight, n_leaves, depth, T):
        # root: [n_leaves, d] — leaf matrix, zero compression
        leaf_indices = torch.arange(
            (1 << (depth - 1)) - 1 if depth > 0 else 0,
            (1 << (depth - 1)) - 1 + n_leaves if depth > 0 else n_leaves,
            device=device
        )
        path_vecs = self.path_enc(leaf_indices, depth)  # [n_leaves, d]
        
        # Select leaf by position + small refinement MLP
        base_leaves = root[:n_leaves]  # [n_leaves, d] — direct matrix access
        refined = self.refine(base_leaves + path_vecs[:n_leaves])  # position-aware refinement
        leaves = refined / (refined.norm(dim=-1, keepdim=True) + 1e-8)
        logits = leaves[:T] @ E_weight.T
        return logits, leaves


# ──── Init ────
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
decoder = FlatPathDecoder(d).to(device)
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

# ──── PHASE 0: Train only E (freeze decoder) ────
print(f"\n{'='*60}")
print(f"PHASE 0: Train E only (freeze decoder, 5 epochs)")
for p in decoder.parameters(): p.requires_grad = False
opt0 = torch.optim.Adam(E.parameters(), lr=0.003)
t0 = time.time()

for epoch in range(5):
    E.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    for bi in range(0, 5000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt0.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            ids = ids[:60]; T = len(ids)
            ids_t = torch.tensor(ids, device=device)
            # Simple echo: E[word] → predict itself via E.weight.T
            emb = E(ids_t)
            logits = emb @ E.weight.T
            loss = F.cross_entropy(logits[:T], ids_t)
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward(); opt0.step()
        ti += 1; tl += (bl / n).item(); tt += 1
    if ti == 0: continue
    avg_loss = tl / ti
    if epoch % 2 == 0 or epoch == 4:
        E_mu = E.weight.mean().item(); E_sigma = E.weight.std().item()
        print(f"  ep {epoch} loss={avg_loss:.4f} E μ={E_mu:.3f} σ={E_sigma:.3f}")

# ──── PHASE 1: Train only decoder (freeze E) ────
print(f"\n{'='*60}")
print(f"PHASE 1: Train decoder only (freeze E, 15 epochs)")
for p in E.parameters(): p.requires_grad = False
for p in decoder.parameters(): p.requires_grad = True
opt1 = torch.optim.Adam(decoder.parameters(), lr=0.003)

for epoch in range(15):
    decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    for bi in range(0, 10000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt1.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            ids = ids[:60]; T = len(ids)
            with torch.no_grad():
                root, mask, n_leaves, depth = heap_encode(E, ids)
            ids_t = torch.tensor(ids, device=device)
            logits, pred_leaves = decoder(root, E.weight, n_leaves, depth, T)
            loss = F.cross_entropy(logits[:T], ids_t) + F.mse_loss(pred_leaves[:T], root[:T])
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward()
        torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]['params'], 2.0); opt1.step()
        ti += 1; tl += (bl / n).item(); tt += 1
    if ti == 0: continue
    avg_loss = tl / ti
    if epoch % 5 == 0 or epoch == 14:
        E.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                ids = ids[:60]; T = len(ids)
                root, mask, n_leaves, depth = heap_encode(E, ids)
                logits, _ = decoder(root, E.weight, n_leaves, depth, T)
                pred = logits[:T].argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T]); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))

        # D1: E self-echo  D2: single-word tree echo
        e_self = 0; e_tot = 0; single_ok = 0; single_tot = 0
        for s, ids in val_data[:30]:
            for wid in ids[:30]:
                ids_t1 = torch.tensor([wid], device=device)
                if (E(ids_t1) @ E.weight.T).argmax(-1).item() == wid: e_self += 1
                e_tot += 1
                root_d2, _, n_leaves_d2, depth_d2 = heap_encode(E, [wid])
                logits_d2, _ = decoder(root_d2, E.weight, n_leaves_d2, depth_d2, 1)
                if logits_d2.argmax(-1).item() == wid: single_ok += 1
                single_tot += 1

        E_mu = E.weight.mean().item(); E_sigma = E.weight.std().item()
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={avg_loss:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% "
              f"E_self={100*e_self/e_tot:.0f}% sing={100*single_ok/single_tot:.0f}% "
              f"E μ={E_mu:.3f} σ={E_sigma:.3f} {elapsed:.0f}s")
        decoder.train()

# ──── PHASE 2: Joint fine-tune (E lr = 0.1 × decoder lr) ────
print(f"\n{'='*60}")
print(f"PHASE 2: Joint fine-tune (E lr=0.0003, decoder lr=0.003)")
for p in E.parameters(): p.requires_grad = True
opt2 = torch.optim.Adam([
    {'params': E.parameters(), 'lr': 0.0003},
    {'params': decoder.parameters(), 'lr': 0.003}
])

for epoch in range(10):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    for bi in range(0, 10000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt2.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            ids = ids[:60]; T = len(ids)
            with torch.no_grad():
                root, mask, n_leaves, depth = heap_encode(E, ids)
            ids_t = torch.tensor(ids, device=device)
            logits, pred_leaves = decoder(root, E.weight, n_leaves, depth, T)
            loss = F.cross_entropy(logits[:T], ids_t) + F.mse_loss(pred_leaves[:T], root[:T])
            bl += loss; n += 1
        if n == 0: continue
        (bl / n).backward()
        torch.nn.utils.clip_grad_norm_(opt2.param_groups[0]['params'], 2.0); opt2.step()
        ti += 1; tl += (bl / n).item(); tt += 1
    if ti == 0: continue
    avg_loss = tl / ti
    if epoch % 3 == 0 or epoch == 9:
        E.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                ids = ids[:60]; T = len(ids)
                root, mask, n_leaves, depth = heap_encode(E, ids)
                logits, _ = decoder(root, E.weight, n_leaves, depth, T)
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
        ids = ids[:60]; T = len(ids)
        root, mask, n_leaves, depth = heap_encode(E, ids)
        logits, _ = decoder(root, E.weight, n_leaves, depth, T)
        pred = logits[:T].argmax(dim=-1).cpu().tolist()
        rf.append(ids[:T]); hp.append(pred)
bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    ids = ids[:60]; T = len(ids)
    with torch.no_grad():
        root, _, _, P, n_leaves, depth = heap_encode(E, ids)
        logits, _ = decoder(root, E.weight, n_leaves, depth, T)
        pred = [id2word.get(p, '?') for p in logits[:T].argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}  [P={P}]")
    print()
