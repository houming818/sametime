"""L1 context-augmented L0 translation: top-K candidates + GRU context re-ranking."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, re, math, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
    def fd(self, x): return self.dp(self.dec(x)[0])

# Load L0 (LaBSE distilled, tree_nce trained)
L0 = nn.Embedding(V, d).to(device)
ckpt_tree = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)
L0.load_state_dict(ckpt_tree['L0']); L0.eval()

# Load L1 (CE autoencode)
L1 = BiGRU(d).to(device)
ckpt_ce = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L1.load_state_dict(ckpt_ce['L1']); L1.eval()

# Build tree (same as tree_nce)
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device)
nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)
if 't_merge' in ckpt_tree: t_merge.load_state_dict(ckpt_tree['t_merge'])

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    return w

def heap_world(tok_ids):
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    w = F.normalize(tw_vec(tok_ids), dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

def pad_to_heap(ids, T):
    k = 1
    while (1 << (k - 1)) < T: k += 1
    nl = 1 << (k - 1)
    return torch.tensor(ids + [0] * (nl - T), device=device), nl

# Pre-compute all ZH world positions (L0) and ZH raw embedding (L1)
print("Pre-computing ZH world positions...")
all_zh_ids = torch.arange(V, device=device)
all_zh_w = F.normalize(heap_world(all_zh_ids), dim=-1)

# L1 ZH embeddings (for context similarity)
all_zh_emb = F.normalize(L0.weight, dim=-1)  # raw embedding, not world pos

# Load test sentences
print("Loading test data...")
test_pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3:
                test_pairs.append((zh_t, en_t))
test_pairs = test_pairs[-2000:]

MAX_LEN = 50
def ng(t, n): return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
def bleu(refs, hyps):
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

# ── L0-only baseline ──
print("\n=== L0-only word-by-word translation ===")
refs, hyps, ct, tt = [], [], 0, 0
for zh_ids, en_ids in random.sample(test_pairs, 100):
    T = min(len(en_ids), len(zh_ids), MAX_LEN)
    en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
    ids_pad = torch.tensor(en_ids, device=device)
    with torch.no_grad():
        en_w = F.normalize(heap_world(ids_pad), dim=-1)
        cos = en_w @ all_zh_w.T
        pred = cos.argmax(-1).tolist()
    refs.append(zh_ids); hyps.append(pred)
    ct += sum(p == z for p, z in zip(pred, zh_ids)); tt += T
print(f"  BLEU: {bleu(refs, hyps):.1f}  Token acc: {100*ct/tt:.1f}%")

# ── L1 context re-ranking (top-K) ──
print(f"\n=== L1 context re-ranking ===\n")
K = 20
lambdas = [0.1, 0.3, 0.5, 1.0]
for lam in lambdas:
    refs, hyps, ct, tt = [], [], 0, 0
    for zh_ids, en_ids in random.sample(test_pairs, 100):
        T = min(len(en_ids), len(zh_ids), MAX_LEN)
        en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
        ids_pad, _ = pad_to_heap(en_ids, T)
        with torch.no_grad():
            emb = L0(ids_pad).unsqueeze(0)
            h_context = L1.fe(emb).squeeze(0)[:T]  # [T, 128]
            en_w = F.normalize(heap_world(ids_pad[:T]), dim=-1)
            l0_cos = en_w @ all_zh_w.T  # [T, 16000]
            topK_vals, topK_idx = l0_cos.topk(K, dim=-1)  # [T, K]
            pred = []
            for t in range(T):
                candidates = topK_idx[t]  # [K]
                # L0 scores (already computed)
                l0_scores = topK_vals[t]
                # L1 context: cos(h_context[t], E_zh[candidate])
                ctx_cos = F.cosine_similarity(
                    h_context[t:t+1], all_zh_emb[candidates], dim=-1)
                final = l0_scores + lam * ctx_cos
                pred.append(candidates[final.argmax()].item())
            refs.append(zh_ids); hyps.append(pred)
            ct += sum(p == z for p, z in zip(pred, zh_ids)); tt += T
    print(f"  λ={lam:.1f}: BLEU={bleu(refs, hyps):.1f}  Token acc={100*ct/tt:.1f}%")

# Show samples with best lambda
print(f"\n=== Samples (λ=0.3) ===")
lam = 0.3
for zh_ids, en_ids in random.sample(test_pairs, 5):
    T = min(len(en_ids), len(zh_ids), 30)
    en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
    ids_pad, _ = pad_to_heap(en_ids, T)
    with torch.no_grad():
        emb = L0(ids_pad).unsqueeze(0)
        h_context = L1.fe(emb).squeeze(0)[:T]
        en_w = F.normalize(heap_world(ids_pad[:T]), dim=-1)
        l0_cos = en_w @ all_zh_w.T
        topK_idx = l0_cos.topk(K, dim=-1)[1]
        pred = []
        for t in range(T):
            candidates = topK_idx[t]
            l0_scores = (en_w[t:t+1] @ all_zh_w[candidates].T).squeeze()
            ctx_cos = F.cosine_similarity(h_context[t:t+1], all_zh_emb[candidates], dim=-1)
            pred.append(candidates[(l0_scores + lam * ctx_cos).argmax()].item())
    print(f"  EN: {sp.decode_ids(en_ids)[:80]}")
    print(f"  ZH: {sp.decode_ids(zh_ids)[:80]}")
    print(f"  PR: {sp.decode_ids(pred)[:80]}")
    print()
