"""Fair comparison: L0 mean embedding accuracy vs LaBSE teacher, same methodology."""
import torch, torch.nn.functional as F, sentencepiece as spm, re
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()

def ok(ids): return all(x != 0 for x in ids)

# Load LaBSE teacher
teacher = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_teacher.pt', map_location=device)['teacher_emb']

# Load our 128D L0 (distilled from LaBSE)
import torch.nn as nn
L0 = nn.Embedding(V, 128).to(device)
ckpt = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_distilled.pt', map_location=device)
L0.load_state_dict(ckpt['L0']); L0.eval()

# Load manual anchors only (not pos-aligned)
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    idx = txt.find("ANCHOR_WORDS = [")
    idx2 = txt.find("]", idx)
    block = txt[idx:idx2+1]
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", block):
        pairs.append((m.group(1), m.group(2)))
print(f"Manual anchors: {len(pairs)}")

valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei, zi, e, z))
print(f"Valid: {len(valid)}")

N = min(len(valid), 500)
valid = valid[:N]

# ── LaBSE teacher (768D) ──
t_en = torch.stack([teacher[torch.tensor(ei, device=device)].mean(dim=0) for ei, zi, _, _ in valid])
t_zh = torch.stack([teacher[torch.tensor(zi, device=device)].mean(dim=0) for zi, ei, _, _ in valid])
t_en = F.normalize(t_en, dim=-1); t_zh = F.normalize(t_zh, dim=-1)
t_cos = F.cosine_similarity(t_en, t_zh, dim=-1)
t_acc = (t_en @ t_zh.T).argmax(-1) == torch.arange(N, device=device)
print(f"\n=== LaBSE teacher (768D mean emb) ===")
print(f"  cos: {t_cos.mean():.4f}  NN acc: {t_acc.float().mean()*100:.1f}%")

# ── Our L0 (128D distilled) ──
s_en = torch.stack([L0.weight[torch.tensor(ei, device=device)].mean(dim=0) for ei, zi, _, _ in valid])
s_zh = torch.stack([L0.weight[torch.tensor(zi, device=device)].mean(dim=0) for zi, ei, _, _ in valid])
s_en = F.normalize(s_en, dim=-1); s_zh = F.normalize(s_zh, dim=-1)
s_cos = F.cosine_similarity(s_en, s_zh, dim=-1)
s_acc = (s_en @ s_zh.T).argmax(-1) == torch.arange(N, device=device)
print(f"\n=== Our L0 (128D mean emb) ===")
print(f"  cos: {s_cos.mean():.4f}  NN acc: {s_acc.float().mean()*100:.1f}%")

# ── Top and bottom pairs ──
print(f"\nTop-10 our L0 pairs:")
top = s_cos.topk(10).indices
for i in top:
    print(f"  {valid[i][2]:15s}→{valid[i][3]:15s} cos={s_cos[i]:.4f}")

print(f"\nBottom-10 our L0 pairs:")
bot = s_cos.topk(10, largest=False).indices
for i in bot:
    print(f"  {valid[i][2]:15s}→{valid[i][3]:15s} cos={s_cos[i]:.4f}")
