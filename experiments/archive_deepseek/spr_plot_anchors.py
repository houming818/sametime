"""Plot anchor pairs on 2D plane via PCA projection of embedding space."""
import torch, torch.nn.functional as F, sentencepiece as spm, sys, numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128

def ok(ids): return all(x != 0 for x in ids)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_alt.pt'

class BiGRU(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = torch.nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.ep = torch.nn.Linear(d, d)
        self.dec = torch.nn.GRU(d, d // 2, bidirectional=True, batch_first=True)
        self.dp = torch.nn.Linear(d, d)
    def fe(self, x):
        return self.ep(self.enc(x)[0])

L0 = torch.nn.Embedding(V, d).to(device)
L1 = BiGRU(d).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()

import re
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    pairs = []
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", txt):
        pairs.append((m.group(1), m.group(2)))

valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei[0], zi[0], e, z))
print(f"Anchor pairs: {len(valid)}")

with torch.no_grad():
    en_ids = torch.tensor([v[0] for v in valid], device=device)
    zh_ids = torch.tensor([v[1] for v in valid], device=device)
    E_en = L0.weight[en_ids]  # [N, 128]
    E_zh = L0.weight[zh_ids]

    # Complex decomposition
    re_en, im_en = E_en[:, :64], E_en[:, 64:]
    re_zh, im_zh = E_zh[:, :64], E_zh[:, 64:]

    # Complex: z = re + i*im.  Complex dot: conj(a) @ b
    complex_en = E_en[:, :64] + 1j * E_en[:, 64:]
    complex_zh = E_zh[:, :64] + 1j * E_zh[:, 64:]

    # Hermitian inner product: <a,b> = sum(conj(a_i) * b_i)
    herm = torch.zeros(len(valid))
    for i in range(len(valid)):
        herm[i] = torch.dot(E_en[i, :64], E_zh[i, :64]) + torch.dot(E_en[i, 64:], E_zh[i, 64:])
        # real part of conj(a)*b = re(a)*re(b) + im(a)*im(b)
    herm_norm = torch.sqrt((E_en**2).sum(-1)) * torch.sqrt((E_zh**2).sum(-1))
    herm_cos = herm / (herm_norm + 1e-8)

    # Simple: project via PCA on [E_en, E_zh] concatenated
    all_emb = torch.cat([E_en.cpu(), E_zh.cpu()]).numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_emb)
    en_2d = all_2d[:len(valid)]
    zh_2d = all_2d[len(valid):]

    # Also compute cosine for coloring
    flat_cos = F.cosine_similarity(E_en, E_zh, dim=-1).cpu().numpy()
    mag_en = torch.sqrt(re_en**2 + im_en**2)
    mag_zh = torch.sqrt(re_zh**2 + im_zh**2)
    mag_cos = F.cosine_similarity(mag_en, mag_zh, dim=-1).cpu().numpy()
    phase_en = torch.atan2(im_en, re_en)
    phase_zh = torch.atan2(im_zh, re_zh)
    phase_cos = F.cosine_similarity(phase_en, phase_zh, dim=-1).cpu().numpy()

    print(f"\n=== PCA variance ratio: {pca.explained_variance_ratio_}")
    print(f"  flat_cos: mean={flat_cos.mean():.4f} std={flat_cos.std():.4f}")
    print(f"  mag_cos:  mean={mag_cos.mean():.4f} std={mag_cos.std():.4f}")
    print(f"  phase_cos:mean={phase_cos.mean():.4f} std={phase_cos.std():.4f}")
    print(f"  herm_cos: mean={herm_cos.mean().item():.4f} std={herm_cos.std().item():.4f}")

    # Count positive vs negative cos
    pos = (flat_cos > 0).sum()
    neg = (flat_cos < 0).sum()
    print(f"  pos_cos: {pos}  neg_cos: {neg}")

    # Output TSV for plotting
    with open('/workspace/anchor_plot.tsv', 'w') as f:
        f.write("en_word\tzh_word\tx_en\ty_en\tx_zh\ty_zh\tflat_cos\tmag_cos\tphase_cos\tcomplex_cos\n")
        for i in range(len(valid)):
            f.write(f"{valid[i][2]}\t{valid[i][3]}\t{en_2d[i][0]:.4f}\t{en_2d[i][1]:.4f}\t{zh_2d[i][0]:.4f}\t{zh_2d[i][1]:.4f}\t{flat_cos[i]:.4f}\t{mag_cos[i]:.4f}\t{phase_cos[i]:.4f}\t{herm_cos[i].item():.4f}\n")

    print(f"\nSaved anchor_plot.tsv")

    # Print top 20 pairs by complex (hermitian) cosine
    sorted_idx = np.argsort(-herm_cos.numpy())[:20]
    print(f"\n=== Top 20 by Hermitian cos ===")
    for i in sorted_idx:
        print(f"  {valid[i][2]:15s}→{valid[i][3]:10s}  flat={flat_cos[i]:.3f} herm={herm_cos[i].item():.3f} mag={mag_cos[i]:.3f}")
