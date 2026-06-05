"""Evaluate L0 word-level translation on sentences — BLEU metric."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, re
import math, sys, random
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

# Load L0 + tree_nce checkpoint
L0 = nn.Embedding(V, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()

# Build tree (depth=5, unit random root, zeros elsewhere)
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device)
nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)
try: t_merge.load_state_dict(ckpt['t_merge'])
except: pass

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

# Pre-compute all ZH world positions
print("Computing all ZH world positions...")
all_zh_ids = torch.arange(V, device=device)
all_zh_w = F.normalize(heap_world(all_zh_ids), dim=-1)  # [16000, 128]

# Load parallel test sentences
print("Loading test data...")
test_pairs = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip())
            en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3:
                test_pairs.append((zh_t, en_t))

# Take last 2000 as test set (not used in anchor training)
test_pairs = test_pairs[-2000:]
print(f"Test sentences: {len(test_pairs)}")

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

# Translate: word-by-word (EN order, ZH vocab)
refs, hyps, correct_tok, total_tok = [], [], 0, 0
for zh_ids, en_ids in random.sample(test_pairs, 200):
    T = min(len(en_ids), len(zh_ids), 50)
    en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
    ids_pad = torch.tensor(en_ids, device=device)
    with torch.no_grad():
        en_w = F.normalize(heap_world(ids_pad), dim=-1)  # [T, 128]
        cos = en_w @ all_zh_w.T  # [T, 16000]
        pred = cos.argmax(-1).tolist()
    refs.append(zh_ids); hyps.append(pred)
    for t in range(T):
        correct_tok += (pred[t] == zh_ids[t])
        total_tok += 1

b = bleu(refs, hyps)
print(f"\n=== L0 word-level translation BLEU ===")
print(f"  BLEU: {b:.1f}")
print(f"  Token acc: {correct_tok}/{total_tok} = {100*correct_tok/total_tok:.1f}%")

# Show samples
print(f"\n=== Samples ===")
for zh_ids, en_ids in random.sample(test_pairs, 5):
    T = min(len(en_ids), len(zh_ids), 30)
    en_ids = en_ids[:T]; zh_ids = zh_ids[:T]
    ids_pad = torch.tensor(en_ids, device=device)
    with torch.no_grad():
        en_w = F.normalize(heap_world(ids_pad), dim=-1)
        cos = en_w @ all_zh_w.T
        pred = cos.argmax(-1).tolist()
    print(f"  EN: {sp.decode_ids(en_ids)[:80]}")
    print(f"  ZH: {sp.decode_ids(zh_ids)[:80]}")
    print(f"  PR: {sp.decode_ids(pred)[:80]}")
    print()
