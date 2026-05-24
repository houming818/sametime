"""
SPR-007 GPU training v2 — Fixed: soft centers, layer routing, semantic hash
Credits: gradient fix + recursive elimination + true baseline by other captain
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

word2id = {"<pad>": 0}
for s in train_sents + val_sents:
    for w in s:
        if w not in word2id: word2id[w] = len(word2id)
V, d = len(word2id), 64

print(f"vocab={V} d={d}")
print("building co-occ embeddings...")
coocc = np.zeros((V, d), dtype=np.float32)
for s in train_sents[:10000]:
    for i, w in enumerate(s):
        if w not in word2id: continue
        wid = word2id[w]
        for j in range(max(0,i-3), min(len(s), i+4)):
            if i != j and s[j] in word2id: coocc[wid, word2id[s[j]] % d] += 1
norms = np.linalg.norm(coocc, axis=1, keepdims=True) + 1e-8
E = torch.tensor(coocc / norms, dtype=torch.float32).cuda()

depth = 8; n_leaves = 1 << depth; n_nodes = n_leaves - 1

# ── FIX 1: Non-recursive layer routing ──
def get_routing_probs(emb, nw):
    """One-shot matrix routing: depth layers of parallel sigmoid gates"""
    N = len(emb)
    probs = torch.ones(N, 1, device=emb.device)
    node_idx = 0
    for layer in range(depth):
        layer_nodes = 1 << layer
        w_layer = nw[node_idx : node_idx + layer_nodes]  # [layer_nodes, d]
        node_idx += layer_nodes
        sc = emb @ w_layer.T  # [N, layer_nodes]
        pr = torch.sigmoid(sc)  # right-turn probability per node
        probs = probs.unsqueeze(2)  # [N, prev_nodes, 1]
        splits = torch.stack([1 - pr, pr], dim=2)  # [N, layer_nodes, 2]
        probs = (probs * splits).view(N, -1)
    return probs  # [N, n_leaves]

# ── FIX 3: True ordered hash with semantic centers ──
def ordered_hash(tokens, emb, cd=0):
    if len(tokens) <= 1:
        return emb[tokens[0]] if len(tokens) == 1 else torch.zeros(emb.shape[1], device=emb.device)
    mid = len(tokens) // 2
    HL = ordered_hash(tokens[:mid], emb, cd + 1)
    HR = ordered_hash(tokens[mid:], emb, cd + 1)
    mask = torch.tensor([1., -1.] * (d // 2 + 1), device=emb.device)[:d]
    return HL + mask * torch.roll(HR, shifts=cd + 1, dims=-1)

# test sentences
test_sents = []
for s in val_sents[:200]:
    ids = [word2id.get(w, 0) for w in s if w in word2id][:16]
    if len(ids) >= 4:
        n = 1
        while n < len(ids): n *= 2
        ids.extend([0] * (n - len(ids)))
        test_sents.append(ids)
    if len(test_sents) >= 50: break

# === Random baseline ===
print(f"\ntest sents={len(test_sents)}")
with torch.no_grad():
    nw_rand = torch.randn(n_nodes, d).cuda() * 0.05
    hashes_rand = torch.stack([ordered_hash(ids, E) for ids in test_sents])
    sim = hashes_rand @ hashes_rand.T / (hashes_rand.norm(dim=1).unsqueeze(1) * hashes_rand.norm(dim=1).unsqueeze(0) + 1e-8)
    disc_rand = sim[torch.eye(len(hashes_rand), device=E.device) == 0].mean().item()
print(f"random node_W: inter-sent sim = {disc_rand:.3f}")

# === Train with soft centers (FIX 2) ===
nw_trained = nn.Parameter(torch.randn(n_nodes, d).cuda() * 0.05)
opt = torch.optim.Adam([nw_trained], lr=0.01)
ideal = V / n_leaves

print("\ntraining node_W on GPU (differentiable)...")
t0 = time.time()
for step in range(301):
    opt.zero_grad()
    assign = get_routing_probs(E, nw_trained)  # [V, n_leaves]
    
    # Soft center: gradient flows through assign, no argmax break
    ls = assign.sum(dim=0) + 1e-8
    lc = assign.T @ E / ls.unsqueeze(1)  # [n_leaves, d]
    token_to_center = assign @ lc  # [V, d] — soft reconstruction
    
    loss_cohesion = F.mse_loss(E, token_to_center)
    loss_balance = ((ls - ideal) ** 2).mean() / ideal
    total_loss = loss_cohesion + 2.0 * loss_balance
    total_loss.backward()
    opt.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            active = len(torch.unique(assign.argmax(dim=1)))
        print(f"  step {step:3d}: loss={total_loss.item():.4f} (coh={loss_cohesion.item():.4f}) | active={active}/{n_leaves}")

print(f"trained in {time.time()-t0:.1f}s")

# === True comparison: semantic centers from trained routing ===
with torch.no_grad():
    assign_f = get_routing_probs(E, nw_trained).argmax(dim=1)
    # FIX 3: use leaf-center embeddings, not raw E
    E_sem = torch.zeros_like(E)
    for i in range(n_leaves):
        mask = assign_f == i
        if mask.any(): E_sem[mask] = E[mask].mean(dim=0)
    
    hashes_trained = torch.stack([ordered_hash(ids, E_sem) for ids in test_sents])
    sim_t = hashes_trained @ hashes_trained.T / (hashes_trained.norm(dim=1).unsqueeze(1) * hashes_trained.norm(dim=1).unsqueeze(0) + 1e-8)
    disc_trained = sim_t[torch.eye(len(hashes_trained), device=E.device) == 0].mean().item()

print(f"\n=== full results ===")
print(f"ordered random sim:   {disc_rand:.3f}")
print(f"ordered trained sim:  {disc_trained:.3f}")
print(f"discrimination delta: {disc_rand - disc_trained:+.3f} (positive = better separation)")
print(f"improved: {'YES' if disc_trained < disc_rand else 'NO'}")
