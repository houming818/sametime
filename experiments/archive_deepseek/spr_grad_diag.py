"""Diagnose: InfoNCE gradient magnitudes on L0.weight and tree nodes."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, sys
import numpy as np, random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128
print(f"device={device}  vocab={V}")

def ok(ids): return all(x != 0 for x in ids)

# Load ANCHORS from file
import re
with open('/workspace/spr_anchor_bridge.py') as f:
    pairs = []
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = nn.Linear(d, d)
        self.dec = nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = nn.Linear(d, d)
    def fe(self, x): return self.ep(self.enc(x)[0])
    def fd(self, x): return self.dp(self.dec(x)[0])

L0 = nn.Embedding(V, d).to(device)
nn.init.normal_(L0.weight, 0, 0.02)
L1 = BiGRU(d).to(device)

HP_DEPTH = 3

merge_mats = nn.ModuleList()
for i in range(HP_DEPTH - 1):
    dim_at_level = d // (2 ** i)
    merge_mats.append(nn.Linear(dim_at_level, dim_at_level).to(device))
    nn.init.eye_(merge_mats[-1].weight); nn.init.zeros_(merge_mats[-1].bias)

t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(HP_DEPTH)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for level in range(HP_DEPTH):
        nidx = torch.clamp(tok_ids // (V // (2 ** level)), 0, (2 ** level) - 1) if level > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[level](nidx)
    return w

def heap_multiply(t_vec, w_vec, level):
    if level == HP_DEPTH:
        return t_vec * w_vec
    sub = t_vec.shape[-1] // 2
    tL, tR = t_vec[..., :sub], t_vec[..., sub:]
    wL, wR = w_vec[..., :sub], w_vec[..., sub:]
    LL = heap_multiply(tL, wL, level + 1)
    RR = heap_multiply(tR, wR, level + 1)
    LR = heap_multiply(tL, wR, level + 1)
    RL = heap_multiply(tR, wL, level + 1)
    return merge_mats[level - 1](torch.cat([LL - RR, LR + RL], dim=-1))

def hp_world_pos(tok_ids):
    t = F.normalize(L0.weight[tok_ids], dim=-1)
    w = F.normalize(tw_vec(tok_ids), dim=-1)
    return heap_multiply(t, w, 1)

# Build anchor ID lists
en_ank_ids = []; zh_ank_ids = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi):
        en_ank_ids.append(ei[0]); zh_ank_ids.append(zi[0])
print(f"Anchors: {len(en_ank_ids)}")

# ─── Run one InfoNCE batch, capture gradients ───
L0.train()
for m in merge_mats: m.train()
for tn in t_nodes: tn.train()

opt = torch.optim.Adam(list(L0.parameters()) + list(merge_mats.parameters()) +
                        [p for tn in t_nodes for p in tn.parameters()], lr=0.003)

en_ids = torch.tensor(en_ank_ids[:200], device=device)  # 200 random anchors
zh_ids = torch.tensor(zh_ank_ids[:200], device=device)
n_pairs = len(en_ids)

opt.zero_grad()
en_w = hp_world_pos(en_ids)
zh_w = hp_world_pos(zh_ids)
logits = F.normalize(en_w, dim=-1) @ F.normalize(zh_w, dim=-1).T / 0.07
loss = F.cross_entropy(logits, torch.arange(n_pairs, device=device))
loss.backward()

print(f"\nInfoNCE loss: {loss.item():.4f}")
print(f"\n=== GRADIENT MAGNITUDES ===")

# Gradient on L0: compare anchor tokens vs non-anchor tokens
l0_grad = L0.weight.grad  # [16000, 128]
ank_norm = l0_grad[en_ids].norm(dim=-1)  # [200]
non_ank_tok = torch.tensor(random.sample([i for i in range(V) if i not in en_ank_ids], 200), device=device)
non_ank_norm = l0_grad[non_ank_tok].norm(dim=-1)
print(f"\nL0.weight gradient norm:")
print(f"  anchor tokens: mean={ank_norm.mean():.6f}  max={ank_norm.max():.6f}  min={ank_norm.min():.6f}")
print(f"  non-anchor:    mean={non_ank_norm.mean():.6f}  max={non_ank_norm.max():.6f}  min={non_ank_norm.min():.6f}")
ratio = ank_norm.mean() / (non_ank_norm.mean() + 1e-10)
print(f"  ratio anchor/non-anchor: {ratio:.1f}x")

# Gradient on tree nodes
print(f"\nTree node gradient norm (mean per node):")
for level in range(HP_DEPTH):
    tn_grad = t_nodes[level].weight.grad
    if tn_grad is not None:
        gn = tn_grad.norm(dim=-1).mean()
        print(f"  level {level} ({2**level} nodes): {gn:.6f}")
    else:
        print(f"  level {level} ({2**level} nodes): None")

# Gradient on merge matrices
print(f"\nMerge matrix gradient norm:")
for i, mm in enumerate(merge_mats):
    gw = mm.weight.grad.norm() if mm.weight.grad is not None else 0
    gb = mm.bias.grad.norm() if mm.bias.grad is not None else 0
    print(f"  merge[{i}] weight={gw:.6f} bias={gb:.6f}")

# Top gradient anchor tokens
print(f"\n=== Top-10 anchor tokens by gradient magnitude ===")
top_idx = ank_norm.topk(10).indices
for rank, idx in enumerate(top_idx):
    tid = en_ids[idx].item()
    tok = sp.id_to_piece(tid)
    grad_n = ank_norm[idx].item()
    print(f"  {rank+1}: {tok:20s}  grad_norm={grad_n:.6f}")
