"""Evaluate LaBSE teacher word-level accuracy with MEAN embeddings (not e[0])."""
import torch, torch.nn.functional as F, sentencepiece as spm, re
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sp = spm.SentencePieceProcessor(); sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()
def ok(ids): return all(x != 0 for x in ids)

teacher = torch.load('/mnt/nas/datasets/wmt17/checkpoints/labse_teacher.pt', map_location=device)['teacher_emb']
print(f"Teacher: {teacher.shape}")

# Load anchors
pairs = []
with open('/workspace/spr_anchor_bridge.py') as f:
    for m in re.finditer(r"\('([^']+)','([^']+)'\)", f.read()):
        pairs.append((m.group(1), m.group(2)))

# Build valid anchors with FULL BPE token lists
valid = []
for e, z in pairs:
    ei = sp.encode_as_ids(e); zi = sp.encode_as_ids(z)
    if ok(ei) and ok(zi): valid.append((ei, zi, e, z))
print(f"Anchor pairs: {len(valid)}")

# Test 1: e[0] only (the buggy way)
en0 = torch.tensor([v[0][0] for v in valid[:500]], device=device)
zh0 = torch.tensor([v[1][0] for v in valid[:500]], device=device)
e0_cos = F.cosine_similarity(teacher[en0], teacher[zh0], dim=-1)
e0_n = F.normalize(teacher[en0], dim=-1)
z0_n = F.normalize(teacher[zh0], dim=-1)
e0_acc = (e0_n @ z0_n.T).argmax(-1) == torch.arange(e0_n.shape[0], device=device)
print(f"\n=== e[0] only (buggy) ===")
print(f"  cos: {e0_cos.mean():.4f}  NN acc: {e0_acc.float().mean()*100:.1f}%")
# Count how many e[0] are the same
same_e0 = (en0.unsqueeze(0) == en0.unsqueeze(1)).sum() - len(en0)
print(f"  duplicate e[0] pairs: {same_e0}")

# Test 2: MEAN embedding across all BPE tokens
en_means = []
zh_means = []
for ei, zi, _, _ in valid[:500]:
    en_means.append(teacher[torch.tensor(ei, device=device)].mean(dim=0))
    zh_means.append(teacher[torch.tensor(zi, device=device)].mean(dim=0))
en_m = F.normalize(torch.stack(en_means), dim=-1)
zh_m = F.normalize(torch.stack(zh_means), dim=-1)
m_cos = F.cosine_similarity(en_m, zh_m, dim=-1)
m_acc = (en_m @ zh_m.T).argmax(-1) == torch.arange(en_m.shape[0], device=device)
print(f"\n=== Mean embedding ===")
print(f"  cos: {m_cos.mean():.4f} std: {m_cos.std():.4f}")
print(f"  range: [{m_cos.min():.4f}, {m_cos.max():.4f}]")
print(f"  positive: {(m_cos>0).sum()}/{len(m_cos)}")
print(f"  NN acc: {m_acc.float().mean()*100:.1f}%")

# Test 3: What are the e[0] token values? Check how many unique tokens
unique_en0 = len(set(v[0][0] for v in valid[:500]))
unique_zh0 = len(set(v[1][0] for v in valid[:500]))
print(f"\n  unique e[0] tokens: {unique_en0}/500")
print(f"  unique z[0] tokens: {unique_zh0}/500")

# Show top duplicate tokens
from collections import Counter
en0_counts = Counter([v[0][0] for v in valid[:500]])
zh0_counts = Counter([v[1][0] for v in valid[:500]])
print(f"  Most common e[0]: {[(sp.id_to_piece(k), v) for k, v in en0_counts.most_common(5)]}")
print(f"  Most common z[0]: {[(sp.id_to_piece(k), v) for k, v in zh0_counts.most_common(5)]}")

# Test 4: mBERT / XLM-R comparison — can we do better?
# Actually just check: what cos do random pairs have?
rand_cos = []
for i in range(100):
    j = (i + 137) % len(en_m)
    rand_cos.append(F.cosine_similarity(en_m[i:i+1], zh_m[j:j+1], dim=-1).item())
rand_cos = torch.tensor(rand_cos)
print(f"\n=== Random pair baseline ===")
print(f"  cos: {rand_cos.mean():.4f} std: {rand_cos.std():.4f}")
