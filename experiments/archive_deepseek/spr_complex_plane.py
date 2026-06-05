"""Complex plane analysis: treat 128D embedding as real[64] + imag[64]."""
import torch, torch.nn.functional as F, sentencepiece as spm, sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V, d = sp.get_piece_size(), 128

def ok(ids): return all(x != 0 for x in ids)

# Load checkpoint from tree_alt or current model
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

L0 = torch.nn.Embedding(V,d).to(device); L1 = BiGRU(d).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()

# Build anchor list from spr_anchor_bridge
import re
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    pairs = []
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", txt):
        pairs.append((m.group(1), m.group(2)))

valid = []
for e,z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei[0], zi[0], e, z))
print(f"Anchor pairs: {len(valid)}")

# Compute complex representations
with torch.no_grad():
    # All EN/ZH anchor embeddings
    en_ids = torch.tensor([v[0] for v in valid], device=device)
    zh_ids = torch.tensor([v[1] for v in valid], device=device)
    E_en = L0.weight[en_ids]  # [N, 128]
    E_zh = L0.weight[zh_ids]

    # Complex: real = first 64, imag = last 64
    re_en, im_en = E_en[:, :64], E_en[:, 64:]
    re_zh, im_zh = E_zh[:, :64], E_zh[:, 64:]

    # Magnitude and phase per token
    mag_en = torch.sqrt(re_en**2 + im_en**2)  # [N, 64]
    mag_zh = torch.sqrt(re_zh**2 + im_zh**2)
    phase_en = torch.atan2(im_en, re_en)
    phase_zh = torch.atan2(im_zh, re_zh)

    # Cosine similarity: flat vs complex components
    flat_cos = F.cosine_similarity(E_en, E_zh, dim=-1)  # [N]
    re_cos = F.cosine_similarity(re_en, re_zh, dim=-1)
    im_cos = F.cosine_similarity(im_en, im_zh, dim=-1)
    mag_cos = F.cosine_similarity(mag_en, mag_zh, dim=-1)
    phase_cos = F.cosine_similarity(phase_en, phase_zh, dim=-1)

    print(f"\n=== Anchor pair complex analysis ===")
    print(f"  flat_cos:   mean={flat_cos.mean():.4f}  std={flat_cos.std():.4f}")
    print(f"  re_cos:     mean={re_cos.mean():.4f}  std={re_cos.std():.4f}")
    print(f"  im_cos:     mean={im_cos.mean():.4f}  std={im_cos.std():.4f}")
    print(f"  mag_cos:    mean={mag_cos.mean():.4f}  std={mag_cos.std():.4f}")
    print(f"  phase_cos:  mean={phase_cos.mean():.4f}  std={phase_cos.std():.4f}")

    # Top-10 and Bottom-10 anchor pairs
    top_idx = flat_cos.topk(10).indices
    bot_idx = flat_cos.topk(10, largest=False).indices

    print(f"\n=== Highest cosine anchor pairs ===")
    for i in top_idx:
        e, z, ew, zw = valid[i]
        print(f"  {ew:15s}→{zw:10s}  flat={flat_cos[i]:.3f} re={re_cos[i]:.3f} im={im_cos[i]:.3f} mag={mag_cos[i]:.3f}")

    print(f"\n=== Lowest cosine anchor pairs ===")
    for i in bot_idx:
        e, z, ew, zw = valid[i]
        print(f"  {ew:15s}→{zw:10s}  flat={flat_cos[i]:.3f} re={re_cos[i]:.3f} im={im_cos[i]:.3f} mag={mag_cos[i]:.3f}")

    # NN test: for each EN anchor, find nearest ZH anchor among ALL 886
    en_norm = F.normalize(E_en, dim=-1)
    zh_norm = F.normalize(E_zh, dim=-1)
    logits = en_norm @ zh_norm.T  # [N, N]
    pred = logits.argmax(-1)
    closed_acc = (pred == torch.arange(len(valid), device=device)).float().mean()
    print(f"\n  Closed-set NN accuracy: {100*closed_acc:.1f}%")
