"""
SPR v4 — EN→ZH Human Balance Test
L0: shared global memory (EN+ZH)
L1: Bi-GRU per-language
L2: identity (cat encoding)
Phase AB: Combined EN+ZH autoencode with repair
Phase C: Zero-param bridge — EN GRU output mean pool → ZH decoder
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random, jieba
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR v4 EN→ZH BALANCE TEST")
print("=" * 60)

# ──── Data ────
print("loading EN-ZH data from NAS...")
zh_en_pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for l in f:
        if "\t" in l:
            zh, en = l.strip().split("\t", 1)
            # Chinese: character-level
            zh_toks = list(jieba.cut(zh.strip()))
            en_toks = en.strip().lower().split()
            if len(zh_toks) >= 2 and len(en_toks) >= 2:
                zh_en_pairs.append((zh_toks, en_toks))

# Also load WMT14 EN for autoencode
print("loading WMT14 EN...")
train_en_sents = []
with open("/data/datasets/wmt14/wmt14.train.de-en") as f:
    for i, l in enumerate(f):
        if i >= 50000: break
        if "\t" in l:
            train_en_sents.append(l.split("\t", 1)[1].strip().lower().split())

val_en_sents = []
with open("/data/datasets/wmt14/wmt14.validation.de-en") as f:
    for i, l in enumerate(f):
        if i >= 300: break
        if "\t" in l: val_en_sents.append(l.split("\t", 1)[1].strip().lower().split())

print(f"EN-ZH: {len(zh_en_pairs)} pairs, EN: {len(train_en_sents)} + {len(val_en_sents)}")

# ──── Shared vocabulary ────
word2id = {"<pad>": 0, "<unk>": 1}
freq = Counter()
for zh, en in zh_en_pairs:
    for w in zh: freq[w] += 1
    for w in en: freq[w] += 1
for s in train_en_sents:
    for w in s: freq[w] += 1
for w, c in freq.most_common():
    if c >= 3: word2id[w] = len(word2id)
for s in val_en_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)

V, d, MAX_LEN = len(word2id), 128, 50
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d}")

# ──── Modules ────
def heap_size(T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    return (1 << k) - 1, 1 << (k - 1), k

def pad_to_heap(ids, T):
    P, n_leaves, depth = heap_size(T)
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

val_data = [(s, [word2id.get(w, 1) for w in s][:MAX_LEN]) for s in val_en_sents[:150] if len(s) >= 2]

# ──── Training set: EN sentences + EN+ZH paired sentences ────
all_sents = []
for s in train_en_sents:
    ids = [word2id.get(w, 1) for w in s[:MAX_LEN]]
    if len(ids) >= 3: all_sents.append(ids)
for zh, en in zh_en_pairs[:50000]:
    ids = [word2id.get(w, 1) for w in en[:MAX_LEN]]
    if len(ids) >= 3: all_sents.append(ids)
    ids_zh = [word2id.get(w, 1) for w in zh[:MAX_LEN]]
    if len(ids_zh) >= 2: all_sents.append(ids_zh)
print(f"training sentences: {len(all_sents)}")

# ══════════════════════════════════════════
# PHASE AB: Combined EN+ZH autoencode
# ══════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE AB: Combined EN+ZH autoencode (repair training, 10 epochs)")
EPOCHS = 10
t0 = time.time()

for epoch in range(EPOCHS):
    L0.train(); L1.train()
    random.shuffle(all_sents)
    ti, tl, tt = 0, 0, 0
    p_drop = 0.25
    for bi in range(0, 5000, 16):
        batch = all_sents[bi:bi + 16]
        if not batch: continue
        opt.zero_grad()
        bl, n_s = torch.tensor(0.0, device=device), 0
        for ids in batch:
            T_orig = min(len(ids), MAX_LEN); ids = ids[:T_orig]
            ids_dropped = []
            kept = 0
            for w in ids:
                if random.random() > p_drop or kept < 2:
                    ids_dropped.append(w); kept += 1
                else:
                    ids_dropped.append(1)  # <unk>
            T_drop = len(ids_dropped)
            ids_pad, n_leaves, depth = pad_to_heap(ids_dropped, T_drop)
            ids_target, _, _ = pad_to_heap(ids, T_orig)
            with torch.no_grad():
                emb = L0(ids_pad).unsqueeze(0)
            ctx = L1.forward_encode(emb)
            leaf_tensor = ctx
            decoded = L1.forward_decode(leaf_tensor)
            logits = decoded.squeeze(0)[:T_orig] @ L0.weight.T
            loss = F.cross_entropy(logits, ids_target[:T_orig])
            bl += loss; n_s += 1
        if n_s == 0: continue
        (bl / n_s).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0); opt.step()
        ti += 1; tl += (bl / n_s).item(); tt += 1
    if ti == 0: continue
    if epoch % 3 == 0 or epoch == EPOCHS - 1:
        L0.eval(); L1.eval(); rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                T = len(ids)
                ids_pad, n_leaves, depth = pad_to_heap(ids, T)
                emb = L0(ids_pad).unsqueeze(0)
                ctx = L1.forward_encode(emb)
                decoded = L1.forward_decode(ctx)
                logits = decoded.squeeze(0) @ L0.weight.T
                pred = logits[:T].argmax(dim=-1).cpu().tolist()
                rf.append(ids[:T]); hp.append(pred)
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r, h in zip(rf, hp) for ri, hi in zip(r, h) if ri == hi) / max(1, sum(len(r) for r in rf))
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={tl / ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% {elapsed:.0f}s")
        L0.train(); L1.train()

# ══════════════════════════════════════════
# PHASE C: L1 raw GRU test — EN hidden states → ZH decoder
# ══════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE C: L1 raw GRU — EN hidden states → ZH decoder (no mean pool)")
L0.eval(); L1.eval()

for i, (zh, en) in enumerate(zh_en_pairs[-20:-5]):
    if len(en) < 3 or len(zh) < 2: continue
    en_ids = [word2id.get(w, 1) for w in en[:MAX_LEN]]
    zh_ids = [word2id.get(w, 1) for w in zh[:MAX_LEN]]
    T_en, T_zh = len(en_ids), len(zh_ids)
    with torch.no_grad():
        en_pad, n_leaves_en, _ = pad_to_heap(en_ids, T_en)
        en_emb = L0(en_pad).unsqueeze(0)
        ctx_en = L1.forward_encode(en_emb)  # [1, n_leaves_en, d]

        # Use raw EN GRU hidden states (not mean pooled), tile to match ZH length
        out_len = max(T_zh, 1)
        ctx_tiled = ctx_en.mean(dim=1, keepdim=True).expand(-1, out_len, -1)  # mean pool → tile

        decoded_zh = L1.forward_decode(ctx_tiled)
        logits_zh = decoded_zh.squeeze(0)[:T_zh] @ L0.weight.T
        pred = [id2word.get(p, '?') for p in logits_zh.argmax(dim=-1).cpu().tolist()]
        print(f"  EN: {' '.join(en[:10])}")
        print(f"  ZH: {''.join(zh[:15])}")
        print(f"  PR: {''.join(pred[:15])}")
        print()

