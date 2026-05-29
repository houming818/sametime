"""
SPR v4 — CBOW-enhanced autoencode + VecMap Bridge
L0: shared BPE vocabulary (16000 subwords)
L1: Bi-GRU (shared across EN+ZH, mixed training)
Phase AB: autoencode + CBOW loss (word-level clustering)
Phase C: cosine similarity diagnosis + repair BLEU test
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, sentencepiece as spm
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR v4 CBOW-enhanced")
print("=" * 60)

# ──── BPE ────
sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
MAX_LEN, d = 50, 128
print(f"BPE vocab={V}")

# ──── Data ────
print("loading data...")
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh, en = l.strip().split("\t", 1)
            zh_toks, en_toks = sp.encode_as_ids(zh.strip()), sp.encode_as_ids(en.strip().lower())
            if len(zh_toks) >= 2 and len(en_toks) >= 2: pairs.append((zh_toks, en_toks))

train_en = []
with open("/data/datasets/wmt14/wmt14.train.de-en") as f:
    for i, l in enumerate(f):
        if i >= 30000: break
        if "\t" in l: train_en.append(sp.encode_as_ids(l.split("\t", 1)[1].strip().lower()))
val_en = [sp.encode_as_ids(l.split("\t", 1)[1].strip().lower()) for l in open("/data/datasets/wmt14/wmt14.validation.de-en")][:300]
print(f"EN-ZH pairs={len(pairs)} EN train={len(train_en)} val={len(val_en)}")

# ──── Modules ────
def heap_size(T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    return 1 << (k - 1), k

def pad_to_heap(ids, T):
    n_leaves, depth = heap_size(T)
    ids_pad = ids + [0] * (n_leaves - T)
    return torch.tensor(ids_pad, device=device), n_leaves, depth

L0 = nn.Embedding(V, d).to(device)
nn.init.normal_(L0.weight, 0, 0.02)

class BiGRUCodec(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.enc_proj = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dec_proj = nn.Linear(d, d)
    def forward_encode(self, x): return self.enc_proj(self.enc(x)[0])
    def forward_decode(self, x): return self.dec_proj(self.dec(x)[0])

L1 = BiGRUCodec(d).to(device)
opt = torch.optim.Adam(list(L0.parameters()) + list(L1.parameters()), lr=0.003)
print(f"params={sum(p.numel() for m in [L0,L1] for p in m.parameters())/1e6:.1f}M")

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

all_sents = []
for ids in train_en[:20000]:
    if len(ids) >= 3: all_sents.append(ids[:MAX_LEN])
for zh, en in pairs[:30000]:
    if len(en) >= 3: all_sents.append(en[:MAX_LEN])
    if len(zh) >= 2: all_sents.append(zh[:MAX_LEN])
val_data = [(ids[:MAX_LEN], ids[:MAX_LEN]) for ids in val_en if len(ids) >= 2]
print(f"training sentences: {len(all_sents)}")

# ══════════════════════════════════════════
# PHASE AB: CBOW-enhanced autoencode
# ══════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE AB: CBOW-enhanced autoencode (repair + word clustering)")
EPOCHS, t0 = 10, time.time()

for epoch in range(EPOCHS):
    L0.train(); L1.train()
    random.shuffle(all_sents)
    ti, tl, tt = 0, 0, 0
    for bi in range(0, 5000, 16):
        batch = all_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl, n_s = torch.tensor(0.0, device=device), 0
        for ids in batch:
            T_orig = min(len(ids), MAX_LEN); ids = ids[:T_orig]
            ids_dropped = [w for w in ids]; kept = 0
            for i in range(len(ids_dropped)):
                if random.random() > 0.25 or kept < 2: kept += 1
                else: ids_dropped[i] = 1
            ids_pad, n_leaves, depth = pad_to_heap(ids_dropped, T_orig)
            ids_target, _, _ = pad_to_heap(ids, T_orig)
            with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
            ctx = L1.forward_encode(emb)
            decoded = L1.forward_decode(ctx)
            logits = decoded.squeeze(0)[:T_orig] @ L0.weight.T
            loss = F.cross_entropy(logits, ids_target[:T_orig])

            # Polish-notation heap-sibling: each pair gets tree-level positional encoding
            h = ctx.squeeze(0)
            n_pairs = T_orig // 2
            if n_pairs >= 2:
                # Polish positional encoding: which depth level these siblings share
                depth_pos = [int(math.log2(t + n_leaves)) for t in range(T_orig)]
                # Sibling pairs: (0,1) share depth D₀, (2,3) share depth D₁, etc.
                even_h = h[0:2*n_pairs:2]; odd_h  = h[1:2*n_pairs:2]
                # Shallow siblings (closer to root) get larger clustering weight
                even_depths = torch.tensor(depth_pos[0:2*n_pairs:2], device=device).float().unsqueeze(-1)
                odd_depths  = torch.tensor(depth_pos[1:2*n_pairs:2], device=device).float().unsqueeze(-1)
                # Depth weighting: deeper siblings (leaves) cluster less strongly
                dw_even = 1.0 / (even_depths + 1); dw_odd = 1.0 / (odd_depths + 1)
                logits_e = (even_h * dw_even) @ L0.weight.T; logits_o = (odd_h * dw_odd) @ L0.weight.T
                loss = loss + 0.1 * F.cross_entropy(logits_e, ids_target[1:2*n_pairs:2])
                loss = loss + 0.1 * F.cross_entropy(logits_o, ids_target[0:2*n_pairs:2])

            bl += loss; n_s += 1
        if n_s == 0: continue
        (bl / n_s).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0); opt.step()
        ti += 1; tl += (bl / n_s).item(); tt += 1
    if ti == 0: continue
    if epoch % 3 == 0 or epoch == EPOCHS - 1:
        L0.eval(); L1.eval(); rf, hp = [], []
        with torch.no_grad():
            for ids, _ in val_data[:30]:
                T = min(len(ids), MAX_LEN); ids_pad, nL, _ = pad_to_heap(ids[:T], T)
                emb = L0(ids_pad).unsqueeze(0)
                ctx = L1.forward_encode(emb)
                decoded = L1.forward_decode(ctx)
                logits = decoded.squeeze(0)[:T] @ L0.weight.T
                pred = logits.argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T]); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={tl / ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% {elapsed:.0f}s")
        L0.train(); L1.train()

# ══════════════════════════════════════════
# PHASE C: Cosine + Repair diagnostics
# ══════════════════════════════════════════
L0.eval(); L1.eval()
print(f"\n{'='*60}")
print("PHASE C: Repair BLEU diagnostic")

print(f"\n=== Repair BLEU (masked autoencode) ===")
rf_repair, hp_repair = [], []
for s in val_en[:50]:
    ids = s[:MAX_LEN]; T = min(len(ids), MAX_LEN); ids = ids[:T]
    ids_broken = [w for w in ids]; kept = 0
    for i in range(len(ids_broken)):
        if random.random() > 0.25 or kept < 2: kept += 1
        else: ids_broken[i] = 1
    with torch.no_grad():
        ids_pad, nL, _ = pad_to_heap(ids_broken, T)
        emb = L0(ids_pad).unsqueeze(0)
        ctx = L1.forward_encode(emb)
        decoded = L1.forward_decode(ctx)
        logits = decoded.squeeze(0)[:T] @ L0.weight.T
        pred = logits.argmax(dim=-1).cpu().tolist()
    rf_repair.append(ids[:T]); hp_repair.append(pred)
print(f"  Repair BLEU: {compute_bleu(rf_repair, hp_repair):.1f}")

print(f"\n=== CBOW vs no-CBOW repair sample ===")
for s in val_en[:3]:
    ids = s[:MAX_LEN]; T = min(len(ids), MAX_LEN); ids = ids[:T]
    ids_brok = [w for w in ids]; kept = 0
    for i in range(len(ids_brok)):
        if random.random() > 0.25 or kept < 2: kept += 1
        else: ids_brok[i] = 1
    with torch.no_grad():
        ids_pad, nL, _ = pad_to_heap(ids_brok, T)
        emb = L0(ids_pad).unsqueeze(0)
        ctx = L1.forward_encode(emb)
        decoded = L1.forward_decode(ctx)
        logits = decoded.squeeze(0)[:T] @ L0.weight.T
        pred = sp.decode_ids(logits.argmax(dim=-1).cpu().tolist())
        gold = sp.decode_ids(ids)
        brok = sp.decode_ids(ids_brok)
        print(f"  broken: {brok[:60]}")
        print(f"  repair: {pred[:60]}")
        print(f"  gold:   {gold[:60]}")
        print()
