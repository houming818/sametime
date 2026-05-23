"""
SPR-007 GPU training — 50K WMT14 sentences
Train node_W with sigmoid routing + cohesion loss on GPU
Compare: random node_W vs trained node_W on ordered hash discrimination
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, time
from collections import Counter

train_file = "/data/datasets/wmt14/wmt14.train.de-en"
val_file = "/data/datasets/wmt14/wmt14.validation.de-en"

def load_sents(path, n):
    sents = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= n: break
            if "\t" in l: sents.append(l.split("\t", 1)[1].strip().lower().split())
    return sents

print("loading data...")
train_sents = load_sents(train_file, 50000)
val_sents = load_sents(val_file, 1000)
print(f"train={len(train_sents)} val={len(val_sents)}")

# vocabulary
word2id = {}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64
id2word = {v:k for k,v in word2id.items()}
print(f"vocab={V} d={d}")

# co-occ embeddings
print("building co-occ embeddings...")
coocc = np.zeros((V, d), dtype=np.float32)
for s in train_sents:
    for i, w in enumerate(s):
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j and s[j] in word2id:
                coocc[wid, word2id[s[j]] % d] += 1
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32).cuda()

# tree
depth = 8; n_leaves = 1<<depth; n_nodes = n_leaves-1

def soft_assign(emb, nw, depth, idx=0):
    if depth==0: return torch.ones(len(emb), 1, device=emb.device)
    sc = emb @ nw[idx]; pr = torch.sigmoid(sc)
    ch = soft_assign(emb, nw, depth-1, 2*idx+1)
    L = ch.shape[1]; a = torch.zeros(len(emb), L*2, device=emb.device)
    a[:,:L] = (1-pr).unsqueeze(1)*ch; a[:,L:] = pr.unsqueeze(1)*ch
    return a

def ordered_hash(tokens, emb, depth=0):
    """Cyclic shift + sign alternation"""
    if len(tokens) <= 1:
        return emb[tokens[0]] if len(tokens)==1 else torch.zeros(d, device=emb.device)
    mid = len(tokens)//2
    HL = ordered_hash(tokens[:mid], emb, depth+1)
    HR = ordered_hash(tokens[mid:], emb, depth+1)
    mask = torch.tensor([1., -1.] * (d//2 + 1), device=emb.device)[:d]
    return HL + mask * torch.roll(HR, shifts=depth+1, dims=-1)

# === random baseline ===
with torch.no_grad():
    nw_rand = torch.randn(n_nodes, d).cuda() * 0.05
    a_rand = soft_assign(E, nw_rand, depth)
    leaf_rand = a_rand.argmax(dim=1)

# prepare test sentences
test_sents = []
for s in val_sents[:200]:
    ids = [word2id[w] for w in s if w in word2id][:16]
    if len(ids) >= 4:
        n = 1
        while n < len(ids): n *= 2
        ids.extend([0]*(n - len(ids)))
        test_sents.append(ids)
    if len(test_sents) >= 50: break

print(f"test sents={len(test_sents)}\n")

# ordered hash discrimination (random routing)
hashes_rand = torch.stack([ordered_hash(ids, E) for ids in test_sents])
sim_rand = (hashes_rand @ hashes_rand.T / (hashes_rand.norm(dim=1).unsqueeze(1) * hashes_rand.norm(dim=1).unsqueeze(0) + 1e-8))
mask = torch.eye(len(hashes_rand), device=E.device) == 0
disc_rand = sim_rand[mask].mean().item()
print(f"random node_W: inter-sent sim = {disc_rand:.3f}")

# === train node_W ===
nw_trained = nn.Parameter(torch.randn(n_nodes, d).cuda() * 0.05)
opt = torch.optim.Adam([nw_trained], lr=0.05)

print("training node_W on GPU...")
t0 = time.time()
all_ids = torch.arange(V, device=E.device)

for step in range(300):
    opt.zero_grad()
    assign = soft_assign(E, nw_trained, depth)
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)
    token_to_center = lc[assign.argmax(dim=1)]
    loss = F.mse_loss(E, token_to_center)
    ideal = V / n_leaves
    loss = loss + 3.0 * ((ls - ideal)**2).mean() / ideal
    loss.backward()
    opt.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            a = soft_assign(E, nw_trained.detach(), depth)
            h = a.argmax(dim=1)
            used = len(torch.unique(h))
        print(f"  step {step:3d}: loss={loss.item():.4f}  active leaves={used}/{n_leaves}")

t1 = time.time()
print(f"trained in {t1-t0:.0f}s")

# ordered hash discrimination (trained routing)
with torch.no_grad():
    hashes_trained = torch.stack([ordered_hash(ids, E) for ids in test_sents])
    sim_trained = (hashes_trained @ hashes_trained.T / (hashes_trained.norm(dim=1).unsqueeze(1) * hashes_trained.norm(dim=1).unsqueeze(0) + 1e-8))
    disc_trained = sim_trained[mask].mean().item()

print(f"\ntrained node_W: inter-sent sim = {disc_trained:.3f}")
print(f"improvement: {disc_rand - disc_trained:+.3f} (lower is more discriminative)")

# also test unordered hash (mean)
def unordered_hash(tokens, emb):
    return emb[tokens].mean(dim=0)

hashes_rand_unord = torch.stack([unordered_hash(ids, E) for ids in test_sents])
disc_rand_unord = (hashes_rand_unord @ hashes_rand_unord.T / (hashes_rand_unord.norm(dim=1).unsqueeze(1) * hashes_rand_unord.norm(dim=1).unsqueeze(0) + 1e-8))
disc_rand_unord = disc_rand_unord[torch.eye(len(hashes_rand_unord), device=E.device) == 0].mean().item()

hashes_trained_unord = torch.stack([unordered_hash(ids, E) for ids in test_sents])
disc_trained_unord = (hashes_trained_unord @ hashes_trained_unord.T / (hashes_trained_unord.norm(dim=1).unsqueeze(1) * hashes_trained_unord.norm(dim=1).unsqueeze(0) + 1e-8))
disc_trained_unord = disc_trained_unord[torch.eye(len(hashes_trained_unord), device=E.device) == 0].mean().item()

print(f"\n=== full results ===")
print(f"random on GPU: voc={V} depth=8 leaves=256 train=2s")
print(f"ordered   random={disc_rand:.3f} → trained={disc_trained:.3f}")
print(f"unordered random={disc_rand_unord:.3f} → trained={disc_trained_unord:.3f}")
print(f"ordered advantage: {disc_rand_unord - disc_rand:.3f} lower (more discriminative)")
print(f"training effect: ordered Δ={disc_rand-disc_trained:+.3f} unordered Δ={disc_rand_unord-disc_trained_unord:+.3f}")
