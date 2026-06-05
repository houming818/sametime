"""
Distill LaBSE embeddings into L0 — warm start for heap architecture.
LaBSE (Language-agnostic BERT Sentence Embedding) produces cross-lingual
aligned token embeddings. We project 768D → 128D and train L0 via MSE.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm, sys, numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}")

sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
print(f"BPE vocab={V}")

# Step 1: Load LaBSE and generate embeddings for all BPE tokens
print("\n=== Loading LaBSE model ===")
from sentence_transformers import SentenceTransformer

labse = SentenceTransformer('sentence-transformers/LaBSE')
labse_dim = labse.get_sentence_embedding_dimension()
print(f"  LaBSE dim={labse_dim}")

# Generate embeddings for all BPE tokens
print("  generating BPE token embeddings (this may take 2-3 minutes)...")
bpe_strings = [sp.id_to_piece(i) for i in range(V)]

# Encode in batches
batch_size = 512
all_emb = []
for bi in range(0, V, batch_size):
    batch = bpe_strings[bi:bi + batch_size]
    emb = labse.encode(batch, convert_to_tensor=True, show_progress_bar=False, device=device)
    all_emb.append(emb)
teacher_emb = torch.cat(all_emb, dim=0)  # [V, 768]
print(f"  teacher embeddings: {teacher_emb.shape}")

# Step 2: Check cross-lingual alignment quality of teacher
print("\n=== Teacher alignment check ===")
def ok(ids): return all(x != 0 for x in ids)
# Load anchor pairs
import re
with open('/workspace/spr_anchor_bridge.py') as f:
    txt = f.read()
    pairs = []
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", txt):
        pairs.append((m.group(1), m.group(2)))

valid = []
for e, z in pairs[:500]:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei[0], zi[0], e, z))

en_ids = torch.tensor([v[0] for v in valid[:100]], device=device)
zh_ids = torch.tensor([v[1] for v in valid[:100]], device=device)
en_emb = teacher_emb[en_ids]
zh_emb = teacher_emb[zh_ids]
teacher_cos = F.cosine_similarity(en_emb, zh_emb, dim=-1).mean().item()
print(f"  Teacher cos (anchor pairs): {teacher_cos:.4f}")

# Check closed-set NN accuracy
en_norm = F.normalize(en_emb, dim=-1)
zh_norm = F.normalize(zh_emb, dim=-1)
logits = en_norm @ zh_norm.T
pred = logits.argmax(-1)
teacher_acc = (pred == torch.arange(len(valid[:100]), device=device)).float().mean().item() * 100
print(f"  Teacher closed-set accuracy: {teacher_acc:.1f}%")

# Step 3: Train L0 to match projected teacher embeddings
d = 128
L0 = nn.Embedding(V, d).to(device)
nn.init.normal_(L0.weight, 0, 0.02)
proj = nn.Linear(labse_dim, d).to(device)
nn.init.normal_(proj.weight, 0, 0.1)

opt = torch.optim.Adam(list(L0.parameters()) + list(proj.parameters()), lr=0.01)
EPOCHS = 200

print(f"\n=== Distilling into L0 ({d}D) ===")
for ep in range(EPOCHS):
    opt.zero_grad()
    student = L0.weight  # [V, 128]
    projected = proj(teacher_emb)  # [V, 128]
    loss = F.mse_loss(student, projected)
    loss.backward()
    opt.step()
    if ep % 50 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            en_s = L0.weight[en_ids]
            zh_s = L0.weight[zh_ids]
            student_cos = F.cosine_similarity(en_s, zh_s, dim=-1).mean().item()
            s_norm = F.normalize(en_s, dim=-1)
            z_norm = F.normalize(zh_s, dim=-1)
            logits_s = s_norm @ z_norm.T
            student_acc = (logits_s.argmax(-1) == torch.arange(len(valid[:100]), device=device)).float().mean().item() * 100
            print(f"  ep {ep:3d} loss={loss.item():.6f} cos={student_cos:.4f} acc={student_acc:.1f}%")

# Save
ckpt = {'L0': L0.state_dict(), 'proj': proj.state_dict(), 'teacher_cos': teacher_cos}
torch.save(ckpt, '/mnt/nas/datasets/wmt17/checkpoints/labse_distilled.pt')
print(f"\nSaved: labse_distilled.pt  teacher_cos={teacher_cos:.4f}")
