"""Diagnose warm-start L0 cos distribution and world_pos accuracy."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, re
device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

L0 = nn.Embedding(V, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/anchor_auto.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()
print(f"Repair BLEU: {ckpt.get('repair_bleu','?'):.1f}")

# Load anchors
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))
valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei[0], zi[0], e, z))
print(f"Anchors: {len(valid)}")

en_ids = torch.tensor([v[0] for v in valid[:500]], device=device)
zh_ids = torch.tensor([v[1] for v in valid[:500]], device=device)

# Raw cos
en_emb = F.normalize(L0.weight[en_ids], dim=-1)
zh_emb = F.normalize(L0.weight[zh_ids], dim=-1)
flat_cos = F.cosine_similarity(en_emb, zh_emb, dim=-1)
print(f"\nRaw L0 cos: mean={flat_cos.mean():.4f} std={flat_cos.std():.4f}")
print(f"  range: [{flat_cos.min():.4f}, {flat_cos.max():.4f}]")
pos = (flat_cos > 0).sum().item()
print(f"  positive: {pos}/{len(flat_cos)}")

# Closed-set NN
logits = en_emb @ zh_emb.T
acc = (logits.argmax(-1) == torch.arange(logits.shape[0], device=device)).float().mean() * 100
print(f"  Raw NN acc: {acc:.1f}%")

# World pos with various tree init
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device)
nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

def tw_vec(ids):
    w = torch.zeros(len(ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(ids)
        w = w + t_nodes[l](nidx)
    return w

def tw_pos(ids):
    t = F.normalize(L0.weight[ids], dim=-1)
    w = F.normalize(tw_vec(ids), dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    return t_merge(torch.cat([tL * wL - tR * wR, tL * wR + tR * wL], -1))

en_w = tw_pos(en_ids); zh_w = tw_pos(zh_ids)
en_w = F.normalize(en_w, dim=-1); zh_w = F.normalize(zh_w, dim=-1)
w_cos = F.cosine_similarity(en_w, zh_w, dim=-1)
print(f"\nWorld cos: mean={w_cos.mean():.4f} std={w_cos.std():.4f}")
logits_w = en_w @ zh_w.T
w_acc = (logits_w.argmax(-1) == torch.arange(logits_w.shape[0], device=device)).float().mean() * 100
print(f"  World NN acc: {w_acc:.1f}%")

# Check world_pos magnitude before normalization
wp_raw = t_merge(torch.cat([
    F.normalize(L0.weight[en_ids], dim=-1)[:, :64] * F.normalize(tw_vec(en_ids), dim=-1)[:, :64] -
    F.normalize(L0.weight[en_ids], dim=-1)[:, 64:] * F.normalize(tw_vec(en_ids), dim=-1)[:, 64:],
    F.normalize(L0.weight[en_ids], dim=-1)[:, :64] * F.normalize(tw_vec(en_ids), dim=-1)[:, 64:] +
    F.normalize(L0.weight[en_ids], dim=-1)[:, 64:] * F.normalize(tw_vec(en_ids), dim=-1)[:, :64]
], -1))
print(f"  World pos raw norm (before normalize): {wp_raw.norm(dim=-1).mean():.4f}")

# Check w magnitude
w_vec = tw_vec(en_ids)
w_raw_norm = w_vec.norm(dim=-1).mean()
print(f"  tw_vec raw norm: {w_raw_norm:.4f}")
w_norm = F.normalize(w_vec, dim=-1).norm(dim=-1).mean()
print(f"  tw_vec normalized norm: {w_norm:.4f}")
