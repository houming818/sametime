"""L1 InfoNCE training — contrastive learning on sentence context."""
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

L0 = nn.Embedding(V, d).to(device); L1 = BiGRU(d).to(device)
ckpt_ce = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt_ce['L0']); L1.load_state_dict(ckpt_ce['L1'])

# Build tree
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device); nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

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

# Load anchors + build index
pairs_anchor = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs_anchor.append((m.group(1), m.group(2)))
anchor_data = []
for e, z in pairs_anchor:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): anchor_data.append((ei, zi))

# Build sentence context index (hash-accelerated)
print("Building sentence index...")
parallel = []
with open('/mnt/nas/datasets/wmt17/train.zh-en') as f:
    for l in f:
        if '\t' in l:
            zh, en = l.strip().split('\t', 1)
            zh_t = sp.encode_as_ids(zh.strip()); en_t = sp.encode_as_ids(en.strip().lower())
            if len(zh_t) >= 3 and len(en_t) >= 3: parallel.append((zh_t, en_t))

MAX_LEN = 50
ank_hash = {}
for ai, (ei, zi) in enumerate(anchor_data):
    ank_hash.setdefault(ei[0], []).append(ai)
anchor_sents = [[] for _ in range(len(anchor_data))]
for si, (zh_s, en_s) in enumerate(parallel[-50000:]):
    if si % 10000 == 0:
        f = sum(1 for a in anchor_sents if len(a) >= 3)
        print(f"  {si}/50000  filled={f}/{len(anchor_data)}")
        if f >= len(anchor_data) * 0.8: break
    for t, tok in enumerate(en_s[:MAX_LEN]):
        if tok not in ank_hash: continue
        for ai in ank_hash[tok]:
            if len(anchor_sents[ai]) >= 3: continue
            ei = anchor_data[ai][0]
            if t + len(ei) <= len(en_s) and en_s[t:t+len(ei)] == ei:
                anchor_sents[ai].append((en_s[:MAX_LEN], t, anchor_data[ai][1]))

# Precompute ZH world positions for all anchors
N = len(anchor_data)
zh_world_all = torch.zeros(N, d, device=device)
with torch.no_grad():
    for ai, (ei, zi) in enumerate(anchor_data):
        zh_w = F.normalize(heap_world(torch.tensor(zi, device=device)).mean(dim=0), dim=-1)
        zh_world_all[ai] = zh_w

# L1 projection (context → world space)
L1_proj = nn.Linear(d, d).to(device); nn.init.eye_(L1_proj.weight); nn.init.zeros_(L1_proj.bias)

# Train L1 + L1_proj with InfoNCE (same as L0)
opt = torch.optim.Adam(list(L1.enc.parameters()) + list(L1.ep.parameters()) + list(L1_proj.parameters()), lr=0.003)

print(f"\n=== L1 InfoNCE training ({N} anchors) ===")
for ep in range(100):
    L1.train(); L1_proj.train()
    random.shuffle(parallel)
    tl, ti = 0, 0
    for bi in range(0, len(parallel), 8):
        batch = parallel[bi:bi+8]
        if not batch: continue
        opt.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        n_l1 = 0
        for zh_s, en_s in batch:
            T = min(len(en_s), MAX_LEN)
            ids_pad, _ = pad_to_heap(en_s[:T], T)
            with torch.no_grad(): emb = L0(ids_pad).unsqueeze(0)
            h_ctx = L1.fe(emb).squeeze(0)[:T]  # [T, 128]
            # Find anchors in this sentence
            for t in range(T):
                if en_s[t] not in ank_hash: continue
                for ai in ank_hash[en_s[t]]:
                    ei = anchor_data[ai][0]
                    if t + len(ei) > T or en_s[t:t+len(ei)] != ei: continue
                    # Found anchor at position t
                    l1_h = F.normalize(L1_proj(h_ctx[t]), dim=-1)
                    # InfoNCE against all ZH world positions
                    logits = l1_h @ zh_world_all.T / 0.07  # [1, N]
                    batch_loss = batch_loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([ai], device=device))
                    n_l1 += 1
                    break
        if n_l1 > 0:
            (batch_loss / n_l1).backward()
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 2.0)
            opt.step()
            tl += (batch_loss / n_l1).item(); ti += 1
    if ep % 10 == 0 or ep == 99:
        print(f"  ep {ep:3d} loss={tl/max(ti,1):.4f}")

# Evaluate: L1 InfoNCE accuracy (closed-set among anchors, same as L0 gold)
print(f"\n=== L1 evaluation ===")
with torch.no_grad():
    correct, total = 0, 0
    for ai, (ei, zi) in enumerate(anchor_data):
        if len(anchor_sents[ai]) == 0: continue
        en_s, pos, _ = random.choice(anchor_sents[ai])
        T = min(len(en_s), MAX_LEN)
        ids_pad, _ = pad_to_heap(en_s[:T], T)
        emb = L0(ids_pad).unsqueeze(0)
        h_ctx = L1.fe(emb).squeeze(0)
        l1_h = F.normalize(L1_proj(h_ctx[pos]), dim=-1)
        logits = l1_h @ zh_world_all.T / 0.07
        pred = logits.argmax().item()
        correct += (pred == ai); total += 1
    print(f"  L1 InfoNCE acc: {correct}/{total} = {100*correct/total:.1f}%")

print("Done")
