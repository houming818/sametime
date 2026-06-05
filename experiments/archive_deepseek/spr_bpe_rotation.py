"""
BPE token decomposition: space rotation vs distance movement
For each GRU transition h[t]→h[t+1], measure:
  rotation = angle(h[t], h[t+1])  → grammar/function words
  distance = ||h[t+1] - h[t]||    → content words
Determine which BPE tokens are "rotators" vs "movers"
"""
import torch, numpy as np, sentencepiece as spm
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  BPE Rotation vs Distance Analysis")
print("=" * 60)

# Load trained model checkpoint would be here — simulate with random for now
# For actual analysis, load the trained L0 + L1 from disk

# Load BPE
sp = spm.SentencePieceProcessor()
sp.load('/mnt/nas/datasets/wmt17/sp_bpe.model')
V = sp.get_piece_size()

# Load sentences
print("loading data...")
pairs = []
with open("/mnt/nas/datasets/wmt17/train.zh-en") as f:
    for i, l in enumerate(f):
        if i >= 5000: break
        if "\t" in l:
            zh, en = l.split("\t", 1)[:2]
            en_ids = sp.encode_as_ids(en.strip().lower())
            if len(en_ids) >= 5:
                pairs.append(en_ids)

# Simple: use raw BPE embeddings to measure rotation vs distance
# Train a minimal model to get meaningful embeddings
import torch.nn as nn

E = nn.Embedding(V, 128).to(device)
nn.init.normal_(E.weight, 0, 0.02)

class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(128, 128)
    def forward(self, x): return self.proj(self.gru(x)[0])

L1 = BiGRU().to(device)
opt = torch.optim.Adam(list(E.parameters()) + list(L1.parameters()), lr=0.01)

# Quick train (autoencode, 2 epochs)
import random
for ep in range(2):
    random.shuffle(pairs)
    for bi in range(0, 2000, 16):
        batch = pairs[bi:bi+16]
        if not batch: continue
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device); n = 0
        for ids in batch:
            ids_t = torch.tensor(ids[:40], device=device)
            emb = E(ids_t).unsqueeze(0)
            h = L1(emb)
            logits = h.squeeze(0) @ E.weight.T
            loss += torch.nn.functional.cross_entropy(logits, ids_t); n += 1
        (loss / n).backward(); opt.step()
    print(f"  ep {ep} trained")

# Analyze rotation vs distance per token
token_dists = defaultdict(list)
token_angles = defaultdict(list)

E.eval(); L1.eval()
with torch.no_grad():
    for ids in pairs[:200]:
        ids_t = torch.tensor(ids[:40], device=device)
        emb = E(ids_t).unsqueeze(0)
        h = L1(emb).squeeze(0)  # [P, 128]
        h = h[:min(len(ids), 40)]  # only real tokens within trained range
        for t in range(len(h) - 1):
            v1 = h[t]; v2 = h[t+1]
            dist = (v2 - v1).norm().item()
            cos = (v1 @ v2) / (v1.norm() * v2.norm() + 1e-8)
            angle = (1.0 - cos.item())  # 1-cos ~ angle with small-angle approx
            token_dists[ids[t]].append(dist)
            token_angles[ids[t]].append(angle)

# Compute per-token average rotation and distance
token_stats = {}
for tid in set(list(token_dists.keys())):
    avg_dist = sum(token_dists[tid]) / len(token_dists[tid])
    avg_angle = sum(token_angles[tid]) / len(token_angles[tid])
    ratio = avg_angle / max(avg_dist, 1e-8)
    piece = sp.id_to_piece(tid)
    token_stats[piece] = (avg_dist, avg_angle, ratio, len(token_dists[tid]))

# Show top rotators (high angle/distance ratio) and movers (low ratio)
sorted_by_ratio = sorted(token_stats.items(), key=lambda x: -x[1][2])

print(f"\n=== Top ROTATORS (grammar words: angle >> distance) ===")
for piece, (dist, angle, ratio, count) in sorted_by_ratio[:20]:
    if count >= 3:
        print(f"  {piece:20s}  dist={dist:.3f}  angle={angle:.3f}  ratio={ratio:.1f}  n={count}")

print(f"\n=== Top MOVERS (content words: distance >> angle) ===")
sorted_by_dist = sorted(token_stats.items(), key=lambda x: -(x[1][0] / max(x[1][1], 1e-8)))
for piece, (dist, angle, ratio, count) in sorted_by_dist[:20]:
    if count >= 3:
        # dist/angle ratio = larger means more distance-oriented
        dr = dist / max(angle, 1e-8)
        print(f"  {piece:20s}  dist={dist:.3f}  angle={angle:.3f}  d/a={dr:.1f}  n={count}")
