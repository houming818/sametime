"""
SPR Padded Complete Binary Tree — No template selection, heap array structure
Pad all sentences to next power-of-2 → fixed complete binary tree
Heap: parent i → left=2i+1, right=2i+2
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR Padded Heap Echo")
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
train_sents = load_sents(train_file, 10000)
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
print(f"vocab={V} d={d}")

# ──── Heap tree functions ────
def next_pow2(n):
    p = 1
    while p < n: p *= 2
    return p

torch.manual_seed(42)
SIGN_MASK = torch.tensor([1., -1.] * (d // 2 + 1), device=device)[:d]

def heap_size(T):
    """Pad to complete binary tree: P = 2^k - 1, leaves = 2^(k-1) >= T"""
    k = 1
    while (1 << (k-1)) < T: k += 1
    leaves = 1 << (k-1)
    total = (1 << k) - 1
    return total, leaves, k

def heap_encode(E, ids):
    T = len(ids)
    P, n_leaves, depth = heap_size(T)
    ids_pad = ids + [0] * (P - T)
    ids_t = torch.tensor(ids_pad, device=device)
    embs = E(ids_t)  # [P, d]

    # Bottom-up: from last parent to root
    hashes = [h.clone() for h in embs]  # P leaf hashes
    for i in range((P - 2) // 2, -1, -1):
        left = 2 * i + 1; right = 2 * i + 2
        if right >= P: continue
        merged = hashes[left] + SIGN_MASK * torch.roll(hashes[right], shifts=1)
        merged = merged / (merged.norm() + 1e-8)
        hashes[i] = merged

    root = hashes[0]
    mask = torch.tensor([1.0 if t < T else 0.0 for t in range(n_leaves)], device=device)
    return root, embs[:n_leaves], mask, P, n_leaves, depth


def heap_decode(decoder, root, P, n_leaves, depth, T):
    """Top-down BFS split: each level splits all current nodes into children."""
    nodes = [root]  # current level nodes
    for lvl in range(depth - 1):
        split = decoder.splits[lvl % decoder.n_splits]
        next_nodes = []
        for node in nodes:
            out = split(node)
            left = out[:d] / (out[:d].norm() + 1e-8)
            right = out[d:] / (out[d:].norm() + 1e-8)
            next_nodes.extend([left, right])
        nodes = next_nodes
    # nodes now has n_leaves elements (all leaves)
    leaf_tensor = torch.stack(nodes, dim=0)  # [n_leaves, d]
    return leaf_tensor[:T]


class HeapDecoder(nn.Module):
    def __init__(self, d, n_splits=3):
        super().__init__()
        self.d = d
        self.n_splits = n_splits
        self.splits = nn.ModuleList([nn.Linear(d, d * 2) for _ in range(n_splits)])

# ──── Init ────
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
decoder = HeapDecoder(d).to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(decoder.parameters()), lr=0.003)
print(f"params={sum(p.numel() for m in [E, decoder] for p in m.parameters())/1e6:.1f}M")

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
print(f"Training Heap Echo: pad to pow2, heap merge, heap split")
print(f"  epochs=30 batch=16 lr=0.003")
t0 = time.time()

for epoch in range(30):
    E.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0

    for bi in range(0, 2000, 16):
        batch = train_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0

        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            T = len(ids)

            # Encode
            with torch.no_grad():
                root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)

            # Decode
            pred_leaves = heap_decode(decoder, root, P, n_leaves, depth, T)

            # Loss: MSE on leaves + direct dot product
            logits = pred_leaves @ E.weight.T  # [T, V]
            ids_t = torch.tensor(ids, device=device)
            loss = F.cross_entropy(logits, ids_t)
            # Also internal node MSE (compare against encoder's gold hashes)
            bl += loss; n += 1

        if n == 0: continue
        (bl / n).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
        opt.step()
        ti += 1; tl += (bl / n).item(); tt += 1

    if ti == 0: continue

    if epoch % 5 == 0 or epoch == 29:
        E.eval(); decoder.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = len(ids)
                root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
                pred_leaves = heap_decode(decoder, root, P, n_leaves, depth, T)
                logits = pred_leaves @ E.weight.T
                pred = logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids); hp.append(pred)

        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={tl / ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% {elapsed:.0f}s")
        E.train(); decoder.train()

# Final
E.eval(); decoder.eval()
rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        T = len(ids)
        root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
        pred_leaves = heap_decode(decoder, root, P, n_leaves, depth, T)
        logits = pred_leaves @ E.weight.T
        pred = logits.argmax(dim=-1).cpu().tolist()
        rf.append(ids); hp.append(pred)

bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s]
    if len(ids) < 3: continue
    T = len(ids)
    with torch.no_grad():
        root, gold_leaves, mask, P, n_leaves, depth = heap_encode(E, ids)
        pred_leaves = heap_decode(decoder, root, P, n_leaves, depth, T)
        logits = pred_leaves @ E.weight.T
        pred = [id2word.get(p, '?') for p in logits.argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}  [P={P}]")
    print()
