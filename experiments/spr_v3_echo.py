"""
SPR v3 Echo — Fractal Decoder validation
Encoder: AttnPool (no fixed templates, pure semantic singularity)
Decoder: ContinuousSplitCell × 5 depth → 32 leaves → tokens
Goal: verify root→leaves works without GRU or fixed templates
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  SPR v3 ECHO — Fractal Decoder Validation")
print("=" * 60)

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

# ──── Data ────
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

V, d, MAX_LEN = len(word2id), 128, 32
MAX_DEPTH = 5; N_LEAVES = 1 << MAX_DEPTH
id2word = {v: k for k, v in word2id.items()}
print(f"vocab={V} d={d} max_len={MAX_LEN} depth={MAX_DEPTH} leaves={N_LEAVES}")

# ──── Modules ────
class AttnPoolEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
    def forward(self, emb):
        attn = F.softmax(emb @ self.query, dim=0)  # [T]
        root = (attn.unsqueeze(-1) * emb).sum(dim=0)
        return root / (root.norm() + 1e-8)

class ContinuousSplitCell(nn.Module):
    def __init__(self, d, max_depth=8):
        super().__init__()
        self.depth_emb = nn.Embedding(max_depth, d)
        self.dir_emb_L = nn.Embedding(max_depth, d)
        self.dir_emb_R = nn.Embedding(max_depth, d)
        self.net = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d * 2))
    def forward(self, h, depth):
        d_idx = torch.tensor(depth, device=h.device)
        d_emb = self.depth_emb(d_idx)
        hL = h + d_emb + self.dir_emb_L(d_idx)
        hR = h + d_emb + self.dir_emb_R(d_idx)
        rawL = self.net(hL)
        rawR = self.net(hR)
        hl, hr = torch.chunk(rawL, 2, dim=-1)[0], torch.chunk(rawR, 2, dim=-1)[1]
        return hl / (hl.norm() + 1e-8), hr / (hr.norm() + 1e-8)

class FractalDecoder(nn.Module):
    def __init__(self, d, V, max_depth=5, n_splits=3):
        super().__init__()
        self.d = d; self.max_depth = max_depth; self.n_leaves = 1 << max_depth
        self.splits = nn.ModuleList([ContinuousSplitCell(d, max_depth) for _ in range(n_splits)])
        self.W_out = nn.Linear(d, V)
    def forward(self, root):
        nodes = [root]
        for depth in range(self.max_depth):
            split = self.splits[depth % len(self.splits)]
            next_nodes = []
            for node in nodes:
                hl, hr = split(node, depth)
                next_nodes.extend([hl, hr])
            nodes = next_nodes
        leaves = torch.stack(nodes, dim=0)
        return self.W_out(leaves)

# ──── Init ────
torch.manual_seed(42)
E = nn.Embedding(V, d).to(device)
nn.init.normal_(E.weight, 0, 0.02)
encoder = AttnPoolEncoder(d).to(device)
decoder = FractalDecoder(d, V, MAX_DEPTH, n_splits=3).to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=0.003)
nP = sum(p.numel() for m in [E, encoder, decoder] for p in m.parameters())
print(f"params={nP/1e6:.1f}M")

# ──── BLEU ────
def ng(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
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
print(f"Training Fractal Echo: AttnPool → root → {MAX_DEPTH} splits → {N_LEAVES} leaves → tokens")
print(f"  epochs=30 batch=16 lr=0.003 sentences=10K")
t0 = time.time()

for epoch in range(10):
    E.train(); encoder.train(); decoder.train()
    random.shuffle(train_sents)
    ti, tl, tt = 0, 0, 0
    
    for bi in range(0, 1000, 16):
        batch = train_sents[bi:bi+16]
        if not batch: continue
        opt.zero_grad()
        bl, n = torch.tensor(0.0, device=device), 0
        
        for s in batch:
            ids = [word2id.get(w, 1) for w in s]
            if len(ids) < 2: continue
            orig_len = min(len(ids), MAX_LEN)
            ids_pad = ids[:MAX_LEN] + [0] * (MAX_LEN - orig_len)
            ids_t = torch.tensor(ids_pad, device=device)
            mask = torch.tensor([1.0 if i < orig_len else 0.0 for i in range(MAX_LEN)], device=device)
            
            emb = E(ids_t)
            root = encoder(emb)
            logits = decoder(root)
            logits_used = logits[:MAX_LEN]
            loss = F.cross_entropy(logits_used, ids_t, ignore_index=0, reduction='sum')
            loss = loss / mask.sum()
            bl += loss; n += 1
        
        if n == 0: continue
        (bl / n).backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0); opt.step()
        ti += 1; tl += (bl / n).item(); tt += 1
    
    if ti == 0: continue
    
    if epoch % 3 == 0 or epoch == 9:
        rf, hp = [], []
        with torch.no_grad():
            for s, ids in val_data[:30]:
                orig_len = len(ids)
                ids_pad = ids[:MAX_LEN] + [0] * (MAX_LEN - min(len(ids), MAX_LEN))
                ids_t = torch.tensor(ids_pad, device=device)
                emb = E(ids_t)
                root = encoder(emb)
                logits = decoder(root)
                pred = logits[:MAX_LEN].argmax(dim=-1).cpu().tolist()
                rf.append(ids[:MAX_LEN])
                hp.append(pred[:MAX_LEN])
        
        bleu = compute_bleu(rf, hp)
        tok_acc = 100 * sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi) / max(1, sum(len(r) for r in rf))
        elapsed = time.time() - t0
        print(f"  ep {epoch:3d} loss={tl/ti:.4f} BLEU={bleu:.1f} tok_acc={tok_acc:.1f}% {elapsed:.0f}s")
        E.train(); encoder.train(); decoder.train()

# ──── Final ────
E.eval(); encoder.eval(); decoder.eval()
rf, hp = [], []
with torch.no_grad():
    for s, ids in val_data:
        orig_len = len(ids)
        ids_pad = ids[:MAX_LEN] + [0] * (MAX_LEN - min(len(ids), MAX_LEN))
        ids_t = torch.tensor(ids_pad, device=device)
        emb = E(ids_t)
        root = encoder(emb)
        logits = decoder(root)
        pred = logits[:MAX_LEN].argmax(dim=-1).cpu().tolist()
        rf.append(ids[:MAX_LEN])
        hp.append(pred[:MAX_LEN])

bleu = compute_bleu(rf, hp)
tok_acc = 100 * sum(1 for r,h in zip(rf,hp) for ri,hi in zip(r,h) if ri==hi) / max(1, sum(len(r) for r in rf))
print(f"\nFinal BLEU-4 = {bleu:.1f}  Token_Accuracy = {tok_acc:.1f}%  Time={time.time()-t0:.0f}s")

print(f"\n=== samples ===")
for i in range(min(5, len(val_sents))):
    s = val_sents[i]; ids = [word2id.get(w, 1) for w in s][:MAX_LEN]
    if len(ids) < 3: continue
    ids_pad = ids + [0] * (MAX_LEN - len(ids))
    ids_t = torch.tensor(ids_pad, device=device)
    with torch.no_grad():
        emb = E(ids_t)
        root = encoder(emb)
        logits = decoder(root)
        pred = [id2word.get(p, '?') for p in logits[:MAX_LEN].argmax(dim=-1).cpu().tolist()]
    src = ' '.join(s[:6])
    hyp = ' '.join(pred[:6])
    print(f"  src: {src}")
    print(f"  hyp: {hyp}")
    print()
