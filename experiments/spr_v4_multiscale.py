"""
SPR v4 Multiscale — Phase AB: combined EN+DE autoencode with repair training
L0: shared global memory (E_de + E_en in same space)
L1: Bi-GRU per language (enc + dec, repair training via token dropout)
L2: identity (cat encoding, no hash compression — BLEU=84.5 proven)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR v4 MULTISCALE — Phase AB")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data ────
print("loading...")
train_pairs = []
with open(train_file) as f:
    for i, l in enumerate(f):
        if i >= 50000: break
        if "\t" in l:
            de, en = l.split("\t")[:2]
            train_pairs.append((de.strip().lower().split(), en.strip().lower().split()))

val_en_sents = []
with open(val_file) as f:
    for i, l in enumerate(f):
        if i >= 500: break
        if "\t" in l: val_en_sents.append(l.split("\t", 1)[1].strip().lower().split())

# Shared vocabulary
word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for de, en in train_pairs:
    for w in de: freq[w] += 1
    for w in en: freq[w] += 1
for w, c in freq.most_common():
    if c >= 2: word2id[w] = len(word2id)
for s in val_en_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d = len(word2id), 128
id2word = {v: k for k, v in word2id.items()}
MAX_LEN = 60
print(f"vocab={V} d={d}")

# ──── Heap sizing ────
def heap_size(T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    leaves = 1 << (k - 1)
    total = (1 << k) - 1
    return total, leaves, k

def pad_to_heap(ids, T):
    P, n_leaves, depth = heap_size(T)
    ids_pad = ids + [0] * (n_leaves - T)
    return torch.tensor(ids_pad, device=device), n_leaves, depth

def drop_rate(epoch):
    return 0.25  # constant 25% dropout for repair training

# ──── L0: shared global memory ────
L0 = nn.Embedding(V, d).to(device)
nn.init.normal_(L0.weight, 0, 0.02)

# ──── L1: Bi-GRU ────
class BiGRUCodec(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.enc_proj = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dec_proj = nn.Linear(d, d)
    def forward_encode(self, x):
        # x: [1, n_leaves, d]
        out, _ = self.enc(x)
        return self.enc_proj(out)  # [1, n_leaves, d]
    def forward_decode(self, x):
        out, _ = self.dec(x)
        return self.dec_proj(out)  # [1, n_leaves, d]

L1 = BiGRUCodec(d).to(device)

# ──── L2: identity (cat encoding) ────
class L2Identity(nn.Module):
    def encode(self, leaf_tensor):
        return leaf_tensor.squeeze(0)  # [n_leaves, d] → root as matrix
    def decode(self, root, n_leaves):
        return root.unsqueeze(0)  # [n_leaves, d] → [1, n_leaves, d]

L2 = L2Identity().to(device)

# ──── Optimizers ────
opt = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=0.003)
print(f"params={sum(p.numel() for m in [L0,L1] for p in m.parameters())/1e6:.1f}M")

# ──── BLEU ────
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

val_data = [(s, [word2id.get(w, 1) for w in s][:MAX_LEN]) for s in val_en_sents[:200] if len(s) >= 2]

# ══════════════════════════════════════════
# PHASE AB: Combined EN+DE autoencode with repair
# ══════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE AB: Combined EN+DE autoencode (repair training, 30 epochs)")
print(f"  epochs=30 batch=16 lr=0.003 dropout_rate=25%")
t0 = time.time()
EPOCHS = 30

all_sents = []
for de, en in train_pairs:
    en_ids = [word2id.get(w, 1) for w in en[:MAX_LEN]]
    de_ids = [word2id.get(w, 1) for w in de[:MAX_LEN]]
    if len(en_ids) >= 3: all_sents.append(en_ids)
    if len(de_ids) >= 3: all_sents.append(de_ids)

print(f"total training sentences: {len(all_sents)} (EN+DE mixed)")

for epoch in range(EPOCHS):
    L0.train(); L1.train()
    random.shuffle(all_sents)
    ti, tl, tt = 0, 0, 0
    p_drop = drop_rate(epoch)

    for bi in range(0, 5000, 16):
        batch = all_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl, n_s = torch.tensor(0.0, device=device), 0
        for ids in batch:
            T_orig = min(len(ids), MAX_LEN)
            ids = ids[:T_orig]

            # Repair: randomly drop 25% of tokens
            ids_dropped = []
            for wid in ids:
                if random.random() > p_drop or len(ids_dropped) < 2:
                    ids_dropped.append(wid)
                else:
                    ids_dropped.append(1)  # <unk> replacement
            T_drop = len(ids_dropped)

            ids_pad, n_leaves, depth = pad_to_heap(ids_dropped, T_drop)
            ids_target, _, _ = pad_to_heap(ids, T_orig)

            with torch.no_grad():
                emb = L0(ids_pad).unsqueeze(0)  # [1, n_leaves, d]

            # L1 encode
            ctx = L1.forward_encode(emb)  # [1, n_leaves, d]

            # L2 identity (cat encoding)
            root = L2.encode(ctx)  # [n_leaves, d]
            leaf_tensor = L2.decode(root, n_leaves)  # [1, n_leaves, d]

            # L1 decode
            decoded = L1.forward_decode(leaf_tensor)  # [1, n_leaves, d]

            # Loss: cross entropy (predicted tokens vs target)
            logits = decoded.squeeze(0)[:T_orig] @ L0.weight.T  # [T_orig, V]
            target = ids_target[:T_orig]
            loss = F.cross_entropy(logits, target)
            bl += loss; n_s += 1

        if n_s == 0: continue
        (bl / n_s).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
        opt.step()
        ti += 1; tl += (bl / n_s).item(); tt += 1

    if ti == 0: continue
    avg_loss = tl / ti

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        L0.eval(); L1.eval()
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = len(ids)
                ids_pad, n_leaves, depth = pad_to_heap(ids, T)
                emb = L0(ids_pad).unsqueeze(0)
                ctx = L1.forward_encode(emb)
                root = L2.encode(ctx)
                leaf_tensor = L2.decode(root, n_leaves)
                decoded = L1.forward_decode(leaf_tensor)
                logits = decoded.squeeze(0) @ L0.weight.T
                pred = logits[:T].argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T]); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        E_mu = L0.weight.mean().item(); E_sigma = L0.weight.std().item()
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={avg_loss:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% "
              f"E μ={E_mu:.3f} σ={E_sigma:.3f} {elapsed:.0f}s")
        L0.train(); L1.train()

# Final
L0.eval(); L1.eval()
rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        T = len(ids)
        ids_pad, n_leaves, depth = pad_to_heap(ids, T)
        emb = L0(ids_pad).unsqueeze(0)
        ctx = L1.forward_encode(emb)
        root = L2.encode(ctx)
        leaf_tensor = L2.decode(root, n_leaves)
        decoded = L1.forward_decode(leaf_tensor)
        logits = decoded.squeeze(0) @ L0.weight.T
        pred = logits[:T].argmax(dim=-1).cpu().tolist()
        rf.append(ids[:T]); hp.append(pred)

bleu_final = compute_bleu(rf, hp)
tok_final = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal: BLEU={bleu_final:.1f} tok_acc={tok_final:.1f}% Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_en_sents))):
    s = val_en_sents[i]; ids = [word2id.get(w, 1) for w in s[:MAX_LEN]]
    if len(ids) < 3: continue
    T = len(ids)
    with torch.no_grad():
        ids_pad, n_leaves, depth = pad_to_heap(ids, T)
        emb = L0(ids_pad).unsqueeze(0)
        ctx = L1.forward_encode(emb)
        root = L2.encode(ctx)
        leaf_tensor = L2.decode(root, n_leaves)
        decoded = L1.forward_decode(leaf_tensor)
        logits = decoded.squeeze(0) @ L0.weight.T
        pred = [id2word.get(p, '?') for p in logits[:T].argmax(dim=-1).cpu().tolist()]
    print(f"  src: {' '.join(s[:6])}")
    print(f"  hyp: {' '.join(pred[:6])}")
    print()
