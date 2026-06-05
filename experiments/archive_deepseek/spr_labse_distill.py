"""Distill LaBSE teacher embeddings into L0 (128D)."""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, re

device = 'cuda' if torch.cuda.is_available() else 'cpu'; d = 128
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

teacher_data = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_teacher.pt', map_location=device)
teacher = teacher_data['teacher_emb']  # [16000, 768]
teacher_dim = teacher.shape[1]
print(f"Teacher: {teacher.shape}")

# Teacher alignment check
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))
valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei[0], zi[0]))
en_ids = torch.tensor([v[0] for v in valid[:500]], device=device)
zh_ids = torch.tensor([v[1] for v in valid[:500]], device=device)
t_cos = F.cosine_similarity(teacher[en_ids], teacher[zh_ids], dim=-1).mean()
tn = F.normalize(teacher[en_ids], dim=-1); zn = F.normalize(teacher[zh_ids], dim=-1)
t_acc = (tn @ zn.T).argmax(-1) == torch.arange(tn.shape[0], device=device)
print(f"Teacher cos: {t_cos:.4f}  NN acc: {t_acc.float().mean()*100:.1f}%")

# Distill
L0 = nn.Embedding(V, d).to(device); nn.init.normal_(L0.weight, 0, 0.02)
proj = nn.Linear(teacher_dim, d).to(device); nn.init.normal_(proj.weight, 0, 0.1)
opt = torch.optim.Adam(list(L0.parameters()) + list(proj.parameters()), lr=0.01)

for ep in range(200):
    opt.zero_grad()
    student = L0.weight
    target = proj(teacher)
    loss = F.mse_loss(student, target)
    loss.backward(); opt.step()
    if ep % 50 == 0 or ep == 199:
        with torch.no_grad():
            s_cos = F.cosine_similarity(L0.weight[en_ids], L0.weight[zh_ids], dim=-1).mean()
            sn = F.normalize(L0.weight[en_ids], dim=-1)
            zn2 = F.normalize(L0.weight[zh_ids], dim=-1)
            s_acc = (sn @ zn2.T).argmax(-1) == torch.arange(sn.shape[0], device=device)
            print(f"  ep {ep:3d} loss={loss.item():.6f} cos={s_cos:.4f} acc={s_acc.float().mean()*100:.1f}%")

torch.save({"L0": L0.state_dict(), "proj": proj.state_dict()},
           '/mnt/nas/datasets/wmt17/checkpoints/labse_distilled.pt')
print("Saved labse_distilled.pt")
