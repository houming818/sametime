"""Visualize anchor pairs in 2D via PCA on world positions (128D heap space)."""
import torch, torch.nn as nn, torch.nn.functional as F, sentencepiece as spm, re, numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

# Load LaBSE distilled L0 + tree_nce structure (random init tree nodes)
L0 = nn.Embedding(V, d).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_distilled.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()

# Build tree (depth=5, random init)
td = 5
t_nodes = nn.ModuleList([nn.Embedding(2 ** i, d).to(device) for i in range(td)])
for tn in t_nodes: nn.init.normal_(tn.weight, 0, 0.1)
rv = torch.randn(1, d, device=device); t_nodes[0].weight.data = rv / rv.norm()
for i in range(1, td): nn.init.zeros_(t_nodes[i].weight)
t_merge = nn.Linear(d, d).to(device); nn.init.eye_(t_merge.weight); nn.init.zeros_(t_merge.bias)

def tw_vec(tok_ids):
    w = torch.zeros(len(tok_ids), d, device=device)
    for l in range(td):
        nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
        w = w + t_nodes[l](nidx)
    return w

def heap_pos(ids_list):
    """World position for mean of multi-BPE-token anchor word."""
    embs = torch.stack([L0.weight[torch.tensor(ids, device=device)].mean(dim=0) for ids in ids_list])
    t = F.normalize(embs, dim=-1)
    w_mean = torch.stack([tw_vec(torch.tensor(ids, device=device)).mean(dim=0) for ids in ids_list])
    w = F.normalize(w_mean, dim=-1)
    tL, tR = t[..., :d // 2], t[..., d // 2:]
    wL, wR = w[..., :d // 2], w[..., d // 2:]
    return t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))

# Load anchor pairs
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))
valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei, zi, e, z))
print(f"Anchor pairs: {len(valid)}")

# Sample 200 pairs for visualization
sample = valid[:200]
en_list = [s[0] for s in sample]
zh_list = [s[1] for s in sample]

with torch.no_grad():
    en_pos = heap_pos(en_list).cpu().numpy()  # [200, 128]
    zh_pos = heap_pos(zh_list).cpu().numpy()

    # PCA via SVD (no sklearn needed)
    all_pos = np.concatenate([en_pos, zh_pos])
    all_pos_tensor = torch.tensor(all_pos).to(device)
    mean = all_pos_tensor.mean(dim=0)
    centered = all_pos_tensor - mean
    U, S, V = torch.svd(centered)
    all_2d = (centered @ V[:, :2]).cpu().numpy()
    var_ratio = (S[:2] ** 2).cpu().numpy() / (S ** 2).sum().cpu().numpy()
    en_2d = all_2d[:200]
    zh_2d = all_2d[200:]
    print(f"PCA variance ratio: {var_ratio}")
    en_norm = F.normalize(torch.tensor(en_pos), dim=-1)
    zh_norm = F.normalize(torch.tensor(zh_pos), dim=-1)
    pair_cos = F.cosine_similarity(en_norm, zh_norm, dim=-1).numpy()

    # Closed-set acc
    logits = en_norm @ zh_norm.T
    acc = (logits.argmax(-1) == torch.arange(200)).float().mean().item() * 100
    print(f"Closed-set acc: {acc:.1f}%")

    # Save TSV
    with open('/workspace/anchor_2d.tsv', 'w') as f:
        f.write("x_en\ty_en\tx_zh\ty_zh\tcos\tlabel\n")
        for i in range(200):
            f.write(f"{en_2d[i][0]:.4f}\t{en_2d[i][1]:.4f}\t{zh_2d[i][0]:.4f}\t{zh_2d[i][1]:.4f}\t{pair_cos[i]:.4f}\t{sample[i][2]}\n")

    print(f"Saved /workspace/anchor_2d.tsv")
    print(f"\nTop-10 high-cos pairs:")
    top = np.argsort(-pair_cos)[:10]
    for i in top:
        print(f"  {sample[i][2]:15s}→{sample[i][3]:15s} cos={pair_cos[i]:.4f}")
